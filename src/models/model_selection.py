"""
Model Selection and Packaging for Production Deployment
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.tracking
import joblib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelSelector:
    """
    Selects best model from MLflow experiments and packages for production
    """

    def __init__(self, mlflow_uri: str = "file:../../models/mlflow_artifacts"):
        """
        Initialize model selector

        Args:
            mlflow_uri: MLflow tracking URI
        """
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()

    def get_best_model(self,
                      experiment_names: list = None,
                      metric: str = "RMSE",
                      ascending: bool = True) -> Dict[str, Any]:
        """
        Find the best model across experiments based on specified metric

        Args:
            experiment_names: List of experiment names to consider
            metric: Metric to optimize (RMSE, MAE, MAPE)
            ascending: True for metrics where lower is better

        Returns:
            Dictionary with best model information
        """
        if experiment_names is None:
            experiment_names = ["GridCast-Baseline-Models", "GridCast-ML-DL-Models"]

        all_runs = []

        # Get runs from all specified experiments
        for exp_name in experiment_names:
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="attribute.status = 'FINISHED'",
                        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
                    )
                    if not runs.empty:
                        runs['experiment_name'] = exp_name
                        all_runs.append(runs)
                else:
                    print(f"⚠️ Experiment '{exp_name}' not found")
            except Exception as e:
                print(f"❌ Error accessing experiment '{exp_name}': {e}")

        if not all_runs:
            raise ValueError("No experiments found with completed runs")

        # Combine all runs
        combined_runs = pd.concat(all_runs, ignore_index=True)

        # Sort by metric
        combined_runs = combined_runs.sort_values(f"metrics.{metric}", ascending=ascending)

        # Get best run
        best_run = combined_runs.iloc[0]

        best_model_info = {
            'run_id': best_run['run_id'],
            'experiment_name': best_run['experiment_name'],
            'model_type': best_run['params.model_type'] if 'params.model_type' in best_run else 'unknown',
            'best_metric_value': best_run[f'metrics.{metric}'],
            'all_metrics': {
                'RMSE': best_run['metrics.RMSE'] if 'metrics.RMSE' in best_run else None,
                'MAE': best_run['metrics.MAE'] if 'metrics.MAE' in best_run else None,
                'MAPE': best_run['metrics.MAPE'] if 'metrics.MAPE' in best_run else None,
            },
            'run_name': best_run['tags.mlflow.runName'] if 'tags.mlflow.runName' in best_run else 'unnamed'
        }

        return best_model_info

    def package_champion_model(self,
                              best_model_info: Dict[str, Any],
                              output_dir: str = "../../models/saved_models") -> str:
        """
        Package the champion model for production deployment

        Args:
            best_model_info: Information about the best model
            output_dir: Directory to save packaged model

        Returns:
            Path to packaged model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        run_id = best_model_info['run_id']
        model_type = best_model_info['model_type']

        print(f"📦 Packaging champion model: {model_type}")
        print(f"   Run ID: {run_id}")
        print(f"   Performance: {best_model_info['all_metrics']}")

        try:
            # Download model artifacts from MLflow
            model_uri = f"runs:/{run_id}/model"
            local_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(output_path / "temp_model")
            )

            # Copy to standard location
            champion_model_path = output_path / "champion_model.joblib"

            if model_type in ['xgboost', 'catboost']:
                # For tree models, copy the model file
                model_files = list(Path(local_model_path).glob("*.joblib")) + \
                             list(Path(local_model_path).glob("*.pkl")) + \
                             list(Path(local_model_path).glob("model.pkl"))

                if model_files:
                    shutil.copy2(model_files[0], champion_model_path)
                else:
                    print("⚠️ No model file found in artifacts")\n
            elif model_type in ['lstm', 'gru']:
                # For PyTorch models
                model_files = list(Path(local_model_path).glob("*.pth")) + \
                             list(Path(local_model_path).glob("*.pt"))

                if model_files:
                    champion_model_path = output_path / "champion_model.pth"
                    shutil.copy2(model_files[0], champion_model_path)

            # Save model metadata
            metadata = {
                'model_type': model_type,
                'run_id': run_id,
                'experiment_name': best_model_info['experiment_name'],
                'performance_metrics': best_model_info['all_metrics'],
                'champion_metric': best_model_info['best_metric_value'],
                'packaging_timestamp': pd.Timestamp.now().isoformat(),
                'model_description': f"Champion {model_type} model for energy demand forecasting"
            }

            metadata_path = output_path / "champion_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Clean up temp directory
            shutil.rmtree(output_path / "temp_model", ignore_errors=True)

            print(f"✅ Champion model packaged successfully")
            print(f"   Model: {champion_model_path}")
            print(f"   Metadata: {metadata_path}")

            return str(champion_model_path)

        except Exception as e:
            print(f"❌ Error packaging model: {e}")
            raise

    def create_model_comparison_report(self,
                                     experiment_names: list = None,
                                     output_dir: str = "../../data/processed") -> str:
        """
        Create comprehensive model comparison report

        Args:
            experiment_names: Experiments to include
            output_dir: Directory to save report

        Returns:
            Path to saved report
        """
        if experiment_names is None:
            experiment_names = ["GridCast-Baseline-Models", "GridCast-ML-DL-Models"]

        all_results = []

        # Collect results from all experiments
        for exp_name in experiment_names:
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="attribute.status = 'FINISHED'"
                    )

                    if not runs.empty:
                        for _, run in runs.iterrows():
                            result = {
                                'model_name': run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else 'unnamed',
                                'model_type': run['params.model_type'] if 'params.model_type' in run else 'unknown',
                                'experiment': exp_name,
                                'run_id': run['run_id'],
                                'RMSE': run['metrics.RMSE'] if 'metrics.RMSE' in run else None,
                                'MAE': run['metrics.MAE'] if 'metrics.MAE' in run else None,
                                'MAPE': run['metrics.MAPE'] if 'metrics.MAPE' in run else None,
                                'MSE': run['metrics.MSE'] if 'metrics.MSE' in run else None,
                            }
                            all_results.append(result)
            except Exception as e:
                print(f"Error processing experiment {exp_name}: {e}")

        if not all_results:
            raise ValueError("No results found in specified experiments")

        # Create DataFrame and sort by RMSE
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('RMSE', ascending=True)

        # Add performance rankings
        results_df['RMSE_rank'] = results_df['RMSE'].rank()
        results_df['MAE_rank'] = results_df['MAE'].rank()
        results_df['MAPE_rank'] = results_df['MAPE'].rank()

        # Calculate improvement over baseline (naive)
        baseline_rmse = results_df[results_df['model_name'].str.contains('Naive', na=False)]['RMSE'].iloc[0] \
                       if any(results_df['model_name'].str.contains('Naive', na=False)) else None

        if baseline_rmse:
            results_df['RMSE_improvement_pct'] = ((baseline_rmse - results_df['RMSE']) / baseline_rmse * 100).round(2)

        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "model_comparison_report.csv"
        results_df.to_csv(report_path, index=False)

        # Create summary
        summary = {
            'total_models_evaluated': len(results_df),
            'best_model': {
                'name': results_df.iloc[0]['model_name'],
                'type': results_df.iloc[0]['model_type'],
                'rmse': results_df.iloc[0]['RMSE'],
                'mae': results_df.iloc[0]['MAE'],
                'mape': results_df.iloc[0]['MAPE']
            },
            'baseline_comparison': {
                'baseline_rmse': baseline_rmse,
                'champion_rmse': results_df.iloc[0]['RMSE'],
                'improvement_pct': results_df.iloc[0]['RMSE_improvement_pct'] if baseline_rmse else None
            }
        }

        summary_path = output_path / "model_selection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Model comparison report created:")
        print(f"   Report: {report_path}")
        print(f"   Summary: {summary_path}")

        return str(report_path)


def main():
    """
    Main model selection and packaging workflow
    """
    print("🔍 GridCast Model Selection & Packaging")
    print("=======================================")

    try:
        # Initialize selector
        selector = ModelSelector()

        # Find best model
        print("\\n1. Finding best model across all experiments...")
        best_model = selector.get_best_model(
            metric="RMSE",
            ascending=True
        )

        print(f"\\n🏆 CHAMPION MODEL SELECTED:")
        print(f"   Model: {best_model['run_name']} ({best_model['model_type']})")
        print(f"   RMSE: {best_model['best_metric_value']:.1f} MW")
        print(f"   Experiment: {best_model['experiment_name']}")

        # Package for production
        print("\\n2. Packaging model for production...")
        model_path = selector.package_champion_model(best_model)

        # Create comparison report
        print("\\n3. Creating model comparison report...")
        report_path = selector.create_model_comparison_report()

        print("\\n✅ MODEL SELECTION COMPLETE!")
        print(f"   Champion model ready at: {model_path}")
        print(f"   Comparison report: {report_path}")

        return model_path, report_path

    except Exception as e:
        print(f"❌ Error in model selection: {e}")
        return None, None


if __name__ == "__main__":
    main()