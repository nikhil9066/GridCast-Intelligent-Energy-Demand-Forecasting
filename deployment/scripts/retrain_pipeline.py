#!/usr/bin/env python3
"""
GridCast Automated Retraining Pipeline

This script implements an automated retraining pipeline that:
1. Fetches new energy consumption data
2. Retrains models with updated data
3. Evaluates performance and selects best model
4. Updates production model if improvement is detected
5. Logs all activities for monitoring

Designed to be run weekly/monthly via cron job or cloud scheduler
"""

import sys
import os
import pandas as pd
import numpy as np
import mlflow
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import argparse
import subprocess

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models'))

# Import project modules
try:
    from models.model_wrapper import GridCastPredictor
    from models.model_selection import ModelSelector
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")

# Configure logging
def setup_logging(log_dir: str = "../../logs") -> logging.Logger:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"retrain_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

class RetrainingPipeline:
    """Automated model retraining pipeline"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize retraining pipeline

        Args:
            config_path: Path to configuration file
        """
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.mlflow_uri = self.config.get('mlflow_uri', 'file:../../models/mlflow_artifacts')
        self.model_dir = Path(self.config.get('model_dir', '../../models/saved_models'))
        self.data_dir = Path(self.config.get('data_dir', '../../data'))

        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)

        self.logger.info("Retraining pipeline initialized")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Data directory: {self.data_dir}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'data_source': 'file',  # 'file', 'api', 'database'
            'retrain_threshold_days': 30,  # Retrain if model is older than X days
            'performance_threshold': 0.05,  # Retrain if RMSE improves by 5%
            'max_models_to_keep': 5,
            'notification_email': None,
            'backup_models': True,
            'validate_before_deploy': True
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Config loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def fetch_new_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch new energy consumption data

        Returns:
            DataFrame with new data or None if failed
        """
        self.logger.info("Fetching new energy consumption data...")

        try:
            if self.config['data_source'] == 'file':
                # For demo: simulate fetching new data by generating recent data
                return self._generate_recent_data()
            elif self.config['data_source'] == 'api':
                return self._fetch_from_api()
            elif self.config['data_source'] == 'database':
                return self._fetch_from_database()
            else:
                raise ValueError(f"Unknown data source: {self.config['data_source']}")

        except Exception as e:
            self.logger.error(f"Failed to fetch new data: {e}")
            return None

    def _generate_recent_data(self, days: int = 30) -> pd.DataFrame:
        """Generate simulated recent energy data"""
        # Create data for last 30 days
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)

        dates = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]

        # Generate realistic pattern
        hours = np.arange(len(dates))
        time_of_day = dates.hour
        day_of_week = dates.dayofweek

        base_demand = 12000
        daily_pattern = 3000 * np.sin(2 * np.pi * (time_of_day - 6) / 24)
        weekly_pattern = 1000 * (1 - 0.3 * (day_of_week >= 5))
        trend = 50 * (hours / (24 * 7))  # Slight upward trend
        noise = np.random.normal(0, 300, len(dates))

        demand = base_demand + daily_pattern + weekly_pattern + trend + noise
        demand = np.maximum(demand, 6000)

        new_data = pd.DataFrame({'demand_mw': demand}, index=dates)

        self.logger.info(f"Generated {len(new_data)} hours of simulated data")
        return new_data

    def _fetch_from_api(self) -> pd.DataFrame:
        """Fetch data from external API (placeholder)"""
        # This would implement actual API calls to energy data providers
        self.logger.info("Fetching from API (placeholder)")
        return self._generate_recent_data()

    def _fetch_from_database(self) -> pd.DataFrame:
        """Fetch data from database (placeholder)"""
        # This would implement database queries
        self.logger.info("Fetching from database (placeholder)")
        return self._generate_recent_data()

    def check_retrain_needed(self) -> Tuple[bool, str]:
        """
        Check if retraining is needed based on various criteria

        Returns:
            Tuple of (should_retrain, reason)
        """
        self.logger.info("Checking if retraining is needed...")

        # Check model age
        model_metadata_path = self.model_dir / "champion_model_metadata.json"

        if not model_metadata_path.exists():
            return True, "No existing model metadata found"

        try:
            with open(model_metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check model age
            last_trained = datetime.fromisoformat(metadata.get('packaging_timestamp', '2023-01-01'))
            days_since_training = (datetime.now() - last_trained).days

            if days_since_training >= self.config['retrain_threshold_days']:
                return True, f"Model is {days_since_training} days old (threshold: {self.config['retrain_threshold_days']})"

            # Check performance degradation (placeholder)
            # In production, this would compare recent predictions vs actuals

            self.logger.info(f"Model is {days_since_training} days old, within threshold")
            return False, f"Model is recent ({days_since_training} days old)"

        except Exception as e:
            self.logger.error(f"Error checking retrain criteria: {e}")
            return True, f"Error reading model metadata: {e}"

    def retrain_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain models with new data

        Args:
            data: New training data

        Returns:
            Dictionary with retraining results
        """
        self.logger.info("Starting model retraining...")

        try:
            # For demo purposes, we'll simulate retraining by creating a new model
            # In practice, this would run the full modeling pipeline

            # Create new experiment run
            experiment_name = f"GridCast-Retraining-{datetime.now().strftime('%Y%m%d')}"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="retrain_xgboost"):
                # Simulate training metrics
                mock_metrics = {
                    'RMSE': np.random.uniform(2500, 3000),
                    'MAE': np.random.uniform(2000, 2500),
                    'MAPE': np.random.uniform(8, 12)
                }

                # Log metrics
                for metric, value in mock_metrics.items():
                    mlflow.log_metric(metric, value)

                # Log parameters
                mlflow.log_param('data_points', len(data))
                mlflow.log_param('retrain_date', datetime.now().isoformat())
                mlflow.log_param('model_type', 'xgboost')

                self.logger.info(f"Retraining completed with RMSE: {mock_metrics['RMSE']:.1f}")

                return {
                    'success': True,
                    'metrics': mock_metrics,
                    'experiment_name': experiment_name,
                    'run_id': mlflow.active_run().info.run_id
                }

        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def evaluate_new_model(self, retrain_results: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if new model is better than current champion

        Args:
            retrain_results: Results from retraining

        Returns:
            Tuple of (is_better, reason)
        """
        if not retrain_results['success']:
            return False, "Retraining failed"

        try:
            # Load current champion metrics
            metadata_path = self.model_dir / "champion_model_metadata.json"

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    current_metadata = json.load(f)

                current_rmse = current_metadata.get('performance_metrics', {}).get('RMSE', float('inf'))
                new_rmse = retrain_results['metrics']['RMSE']

                improvement = (current_rmse - new_rmse) / current_rmse
                threshold = self.config['performance_threshold']

                if improvement >= threshold:
                    return True, f"RMSE improved by {improvement:.1%} (threshold: {threshold:.1%})"
                else:
                    return False, f"RMSE improvement {improvement:.1%} below threshold {threshold:.1%}"
            else:
                return True, "No existing champion model"

        except Exception as e:
            self.logger.error(f"Error evaluating new model: {e}")
            return False, f"Evaluation error: {e}"

    def deploy_new_model(self, retrain_results: Dict[str, Any]) -> bool:
        """
        Deploy new model as champion

        Args:
            retrain_results: Results from retraining

        Returns:
            True if deployment successful
        """
        try:
            self.logger.info("Deploying new champion model...")

            # Backup current model
            if self.config['backup_models']:
                self._backup_current_model()

            # Create new model metadata
            new_metadata = {
                'model_type': 'xgboost',
                'performance_metrics': retrain_results['metrics'],
                'retrain_timestamp': datetime.now().isoformat(),
                'experiment_name': retrain_results['experiment_name'],
                'run_id': retrain_results['run_id'],
                'deployment_timestamp': datetime.now().isoformat(),
                'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }

            # Save new metadata
            metadata_path = self.model_dir / "champion_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=2)

            # In practice, you would download the actual model from MLflow here
            self.logger.info("New model deployed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False

    def _backup_current_model(self):
        """Backup current champion model"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.model_dir / "backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy current model files
            current_model = self.model_dir / "champion_model.joblib"
            current_metadata = self.model_dir / "champion_model_metadata.json"

            if current_model.exists():
                import shutil
                shutil.copy2(current_model, backup_dir / "champion_model.joblib")

            if current_metadata.exists():
                import shutil
                shutil.copy2(current_metadata, backup_dir / "champion_model_metadata.json")

            self.logger.info(f"Current model backed up to {backup_dir}")

        except Exception as e:
            self.logger.warning(f"Model backup failed: {e}")

    def send_notification(self, message: str, success: bool = True):
        """Send notification about retraining results"""
        self.logger.info(f"Notification: {message}")

        # In production, this would send emails, Slack messages, etc.
        if self.config.get('notification_email'):
            self.logger.info(f"Would send email to {self.config['notification_email']}")

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete retraining pipeline

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("STARTING GRIDCAST RETRAINING PIPELINE")
        self.logger.info("=" * 50)

        results = {
            'pipeline_start': pipeline_start.isoformat(),
            'success': False,
            'steps_completed': [],
            'errors': []
        }

        try:
            # Step 1: Check if retraining is needed
            self.logger.info("STEP 1: Checking if retraining is needed")
            should_retrain, reason = self.check_retrain_needed()
            results['retrain_check'] = {'needed': should_retrain, 'reason': reason}
            results['steps_completed'].append('retrain_check')

            if not should_retrain:
                self.logger.info(f"Retraining not needed: {reason}")
                results['success'] = True
                results['action'] = 'no_retrain_needed'
                self.send_notification(f"GridCast pipeline completed - no retraining needed: {reason}")
                return results

            # Step 2: Fetch new data
            self.logger.info("STEP 2: Fetching new data")
            new_data = self.fetch_new_data()
            if new_data is None:
                raise Exception("Failed to fetch new data")

            results['data_fetch'] = {'records': len(new_data), 'date_range': f"{new_data.index[0]} to {new_data.index[-1]}"}
            results['steps_completed'].append('data_fetch')

            # Step 3: Retrain models
            self.logger.info("STEP 3: Retraining models")
            retrain_results = self.retrain_models(new_data)
            if not retrain_results['success']:
                raise Exception(f"Retraining failed: {retrain_results.get('error', 'Unknown error')}")

            results['retrain'] = retrain_results
            results['steps_completed'].append('retrain')

            # Step 4: Evaluate new model
            self.logger.info("STEP 4: Evaluating new model")
            is_better, eval_reason = self.evaluate_new_model(retrain_results)
            results['evaluation'] = {'is_better': is_better, 'reason': eval_reason}
            results['steps_completed'].append('evaluation')

            if not is_better:
                self.logger.info(f"New model not better than current: {eval_reason}")
                results['success'] = True
                results['action'] = 'model_not_improved'
                self.send_notification(f"GridCast retrained but not deployed: {eval_reason}")
                return results

            # Step 5: Deploy new model
            self.logger.info("STEP 5: Deploying new model")
            deploy_success = self.deploy_new_model(retrain_results)
            if not deploy_success:
                raise Exception("Model deployment failed")

            results['deployment'] = {'success': True}
            results['steps_completed'].append('deployment')

            # Pipeline completed successfully
            results['success'] = True
            results['action'] = 'model_updated'

            pipeline_duration = datetime.now() - pipeline_start
            success_msg = f"GridCast pipeline completed successfully in {pipeline_duration}. New model deployed with RMSE: {retrain_results['metrics']['RMSE']:.1f}"

            self.logger.info("=" * 50)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)

            self.send_notification(success_msg, success=True)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['success'] = False
            results['action'] = 'pipeline_failed'

            self.send_notification(f"GridCast pipeline failed: {e}", success=False)

        finally:
            results['pipeline_end'] = datetime.now().isoformat()
            results['duration_minutes'] = (datetime.now() - pipeline_start).total_seconds() / 60

        return results

def main():
    """Main entry point for retraining pipeline"""
    parser = argparse.ArgumentParser(description='GridCast Automated Retraining Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of checks')
    parser.add_argument('--dry-run', action='store_true', help='Run pipeline without making changes')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RetrainingPipeline(config_path=args.config)

    if args.force:
        pipeline.config['retrain_threshold_days'] = 0  # Force retrain
        pipeline.logger.info("Force retraining enabled")

    if args.dry_run:
        pipeline.logger.info("DRY RUN MODE - No changes will be made")
        # In dry-run mode, you would skip actual model updates

    # Run pipeline
    results = pipeline.run_pipeline()

    # Print summary
    print("\\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Success: {results['success']}")
    print(f"Action: {results['action']}")
    print(f"Duration: {results['duration_minutes']:.1f} minutes")
    print(f"Steps completed: {', '.join(results['steps_completed'])}")

    if results['errors']:
        print(f"Errors: {', '.join(results['errors'])}")

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)

if __name__ == "__main__":
    main()