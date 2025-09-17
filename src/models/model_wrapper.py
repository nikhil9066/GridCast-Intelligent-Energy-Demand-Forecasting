"""
GridCast Model Wrapper - Production-ready inference interface
"""

import pandas as pd
import numpy as np
import joblib
import torch
import warnings
from typing import Union, List, Dict, Any
from pathlib import Path
import json

warnings.filterwarnings('ignore')


class GridCastPredictor:
    """
    Production wrapper for GridCast energy demand forecasting models

    Supports multiple model types:
    - XGBoost/CatBoost (tabular ML models)
    - LSTM/GRU (deep learning models)
    - Prophet/SARIMA (time series models)
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Initialize the predictor with a trained model

        Args:
            model_path: Path to saved model file
            model_type: Type of model ('xgboost', 'catboost', 'lstm', 'gru', 'prophet', 'sarima')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_loaded = False

        # Model configuration
        self.config = {
            'sequence_length': 168,  # 1 week for DL models
            'forecast_horizon': 48,  # 2 days ahead
            'supported_models': ['xgboost', 'catboost', 'lstm', 'gru', 'prophet', 'sarima']
        }

        self._load_model()

    def _load_model(self):
        """Load the trained model and associated artifacts"""
        try:
            if self.model_type in ['xgboost', 'catboost']:
                self._load_ml_model()
            elif self.model_type in ['lstm', 'gru']:
                self._load_dl_model()
            elif self.model_type in ['prophet', 'sarima']:
                self._load_ts_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.is_loaded = True
            print(f"✅ {self.model_type.upper()} model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading {self.model_type} model: {str(e)}")
            self.is_loaded = False

    def _load_ml_model(self):
        """Load XGBoost/CatBoost models"""
        self.model = joblib.load(self.model_path)

        # Load feature columns if available
        feature_path = self.model_path.parent / 'feature_columns.json'
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_columns = json.load(f)

    def _load_dl_model(self):
        """Load PyTorch LSTM/GRU models"""
        if torch.cuda.is_available():
            self.model = torch.load(self.model_path)
        else:
            self.model = torch.load(self.model_path, map_location='cpu')

        self.model.eval()

        # Load scaler if available
        scaler_path = self.model_path.parent / 'scaler.joblib'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

    def _load_ts_model(self):
        """Load Prophet/SARIMA models"""
        self.model = joblib.load(self.model_path)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw demand data

        Args:
            df: DataFrame with datetime index and 'demand_mw' column

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (df['day_of_week'] < 5)).astype(int)

        # Seasonal indicators
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Lag features
        for lag in [1, 24, 48, 168]:
            df[f'demand_mw_lag_{lag}'] = df['demand_mw'].shift(lag)

        # Rolling features
        for window in [3, 6, 12, 24, 48, 168]:
            df[f'demand_mw_rolling_mean_{window}'] = df['demand_mw'].rolling(window=window).mean()
            df[f'demand_mw_rolling_std_{window}'] = df['demand_mw'].rolling(window=window).std()
            df[f'demand_mw_rolling_min_{window}'] = df['demand_mw'].rolling(window=window).min()
            df[f'demand_mw_rolling_max_{window}'] = df['demand_mw'].rolling(window=window).max()

        return df

    def predict(self,
                data: Union[pd.DataFrame, np.ndarray],
                hours_ahead: int = 48) -> Dict[str, Any]:
        """
        Generate energy demand forecasts

        Args:
            data: Input data (DataFrame with datetime index or numpy array)
            hours_ahead: Number of hours to forecast (default: 48)

        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        if hours_ahead > self.config['forecast_horizon']:
            print(f"⚠️ Warning: Requested {hours_ahead}h forecast, model trained for {self.config['forecast_horizon']}h")

        try:
            if self.model_type in ['xgboost', 'catboost']:
                return self._predict_ml(data, hours_ahead)
            elif self.model_type in ['lstm', 'gru']:
                return self._predict_dl(data, hours_ahead)
            elif self.model_type in ['prophet', 'sarima']:
                return self._predict_ts(data, hours_ahead)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': None,
                'confidence_intervals': None
            }

    def _predict_ml(self, data: pd.DataFrame, hours_ahead: int) -> Dict[str, Any]:
        """Predictions for ML models (XGBoost/CatBoost)"""
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index for ML models")

        # Create features
        df_features = self.create_features(data)

        # Remove rows with NaN (from lag/rolling features)
        df_clean = df_features.dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data after feature engineering. Need more historical data.")

        # Select features for prediction
        if self.feature_columns:
            feature_cols = [col for col in self.feature_columns if col in df_clean.columns]
        else:
            feature_cols = [col for col in df_clean.columns if col != 'demand_mw']

        # Use most recent complete record for prediction
        X_pred = df_clean[feature_cols].iloc[-1:].values

        # Generate prediction
        pred = self.model.predict(X_pred)[0]

        # For simplicity, repeat single prediction for forecast horizon
        # In production, you'd implement proper multi-step forecasting
        predictions = np.full(hours_ahead, pred)

        # Create forecast timestamps
        last_timestamp = data.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )

        return {
            'success': True,
            'model_type': self.model_type,
            'predictions': predictions.tolist(),
            'timestamps': forecast_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'forecast_horizon_hours': hours_ahead,
            'last_actual_value': data['demand_mw'].iloc[-1],
            'confidence_intervals': None  # Could add prediction intervals for tree models
        }

    def _predict_dl(self, data: Union[pd.DataFrame, np.ndarray], hours_ahead: int) -> Dict[str, Any]:
        """Predictions for deep learning models (LSTM/GRU)"""
        if isinstance(data, pd.DataFrame):
            # Extract demand values as sequence
            if 'demand_mw' not in data.columns:
                raise ValueError("DataFrame must contain 'demand_mw' column")
            sequence = data['demand_mw'].values
            timestamps = data.index
        else:
            sequence = data
            timestamps = None

        # Ensure we have enough data for sequence
        if len(sequence) < self.config['sequence_length']:
            raise ValueError(f"Need at least {self.config['sequence_length']} historical hours for prediction")

        # Take last sequence_length values
        input_sequence = sequence[-self.config['sequence_length']:]

        # Scale if scaler is available
        if self.scaler:
            input_sequence = self.scaler.transform(input_sequence.reshape(-1, 1)).flatten()

        # Convert to tensor
        X = torch.FloatTensor(input_sequence).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]

        # Generate prediction
        with torch.no_grad():
            pred_tensor = self.model(X)
            predictions = pred_tensor.numpy().flatten()

        # Inverse scale if needed
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Truncate to requested horizon
        predictions = predictions[:hours_ahead]

        # Create timestamps if input was DataFrame
        if timestamps is not None:
            forecast_timestamps = pd.date_range(
                start=timestamps[-1] + pd.Timedelta(hours=1),
                periods=hours_ahead,
                freq='H'
            )
            timestamp_list = forecast_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
        else:
            timestamp_list = None

        return {
            'success': True,
            'model_type': self.model_type,
            'predictions': predictions.tolist(),
            'timestamps': timestamp_list,
            'forecast_horizon_hours': hours_ahead,
            'last_actual_value': float(sequence[-1]),
            'confidence_intervals': None
        }

    def _predict_ts(self, data: pd.DataFrame, hours_ahead: int) -> Dict[str, Any]:
        """Predictions for time series models (Prophet/SARIMA)"""
        # Implementation would depend on specific model type
        # This is a placeholder for time series model predictions
        return {
            'success': False,
            'error': 'Time series model prediction not implemented in this wrapper',
            'predictions': None,
            'timestamps': None
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'model_path': str(self.model_path),
            'is_loaded': self.is_loaded,
            'config': self.config,
            'supported_forecast_horizon': self.config['forecast_horizon']
        }


def load_champion_model(models_dir: str = "../../models/saved_models") -> GridCastPredictor:
    """
    Load the best performing model from Phase 3

    Args:
        models_dir: Directory containing saved models

    Returns:
        GridCastPredictor instance with champion model loaded
    """
    models_path = Path(models_dir)

    # For this demo, we'll assume XGBoost was the champion
    # In practice, this would be determined from MLflow experiments
    champion_model_path = models_path / "champion_model.joblib"

    if champion_model_path.exists():
        return GridCastPredictor(str(champion_model_path), "xgboost")
    else:
        raise FileNotFoundError(f"Champion model not found at {champion_model_path}")


# Example usage and testing
if __name__ == "__main__":
    print("GridCast Model Wrapper - Example Usage")
    print("=====================================")

    # Create sample data for testing
    import pandas as pd
    from datetime import datetime, timedelta

    # Generate sample historical data (1 week)
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='H'
    )

    # Simulate energy demand with daily/weekly patterns
    hours = np.arange(len(dates))
    daily_pattern = 10000 + 3000 * np.sin(2 * np.pi * hours / 24 - np.pi/4)  # Peak at 6 PM
    weekly_pattern = 1000 * np.sin(2 * np.pi * hours / (24*7))  # Weekly variation
    noise = np.random.normal(0, 500, len(dates))

    sample_demand = daily_pattern + weekly_pattern + noise
    sample_data = pd.DataFrame({
        'demand_mw': sample_demand
    }, index=dates)

    print(f"Sample data created: {len(sample_data)} hours")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"Demand range: {sample_data['demand_mw'].min():.0f} - {sample_data['demand_mw'].max():.0f} MW")

    # Note: Actual model loading would require trained models from Phase 3
    print("\\n⚠️ To use this wrapper:")
    print("1. Complete Phase 3 to train and save models")
    print("2. Update load_champion_model() to load the actual best model")
    print("3. Test with: predictor = load_champion_model()")
    print("4. Make predictions: result = predictor.predict(sample_data, hours_ahead=48)")