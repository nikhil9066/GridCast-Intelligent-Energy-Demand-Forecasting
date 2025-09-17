"""
GridCast Flask API - Production deployment for energy demand forecasting
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
import traceback

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

try:
    from model_wrapper import GridCastPredictor
except ImportError as e:
    print(f"Warning: Could not import model_wrapper: {e}")
    GridCastPredictor = None

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
predictor = None
model_info = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GridCast Energy Demand Forecasting API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
        .endpoint { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #27ae60; }
        .method { font-weight: bold; color: #27ae60; }
        .url { font-family: monospace; background: #f8f8f8; padding: 3px 6px; border-radius: 3px; }
        .example { background: #f8f8f8; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; margin: 10px 0; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.healthy { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f1aeb5; }
        .footer { text-align: center; margin-top: 30px; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .metric { text-align: center; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ GridCast Energy Demand Forecasting API</h1>
            <p style="text-align: center; margin: 0; font-size: 18px;">
                Intelligent 48-hour energy demand forecasting for grid operators
            </p>
        </div>

        <!-- Status Section -->
        <div class="section">
            <h2>🔍 System Status</h2>
            <div class="status {{ status_class }}">
                <strong>API Status:</strong> {{ api_status }}<br>
                <strong>Model:</strong> {{ model_status }}<br>
                <strong>Last Updated:</strong> {{ timestamp }}
            </div>

            {% if model_info %}
            <h3>📊 Model Information</h3>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{{ model_info.model_type }}</td></tr>
                <tr><td>Forecast Horizon</td><td>{{ model_info.supported_forecast_horizon }} hours</td></tr>
                <tr><td>Status</td><td>{{ 'Loaded' if model_info.is_loaded else 'Not Loaded' }}</td></tr>
            </table>
            {% endif %}
        </div>

        <!-- API Endpoints -->
        <div class="section">
            <h2>🚀 API Endpoints</h2>

            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/health</div>
                <p>Check API and model health status</p>
                <div class="example">curl -X GET "{{ base_url }}/health"</div>
            </div>

            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/forecast</div>
                <p>Generate energy demand forecast</p>
                <strong>Parameters:</strong>
                <ul>
                    <li><code>hours_ahead</code> (optional): Number of hours to forecast (default: 48, max: 48)</li>
                    <li><code>data</code> (optional): Historical demand data as JSON array</li>
                </ul>
                <div class="example">curl -X POST "{{ base_url }}/forecast" \\
  -H "Content-Type: application/json" \\
  -d '{"hours_ahead": 24}'</div>
            </div>

            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/forecast/batch</div>
                <p>Generate multiple forecasts with different horizons</p>
                <div class="example">curl -X POST "{{ base_url }}/forecast/batch" \\
  -H "Content-Type: application/json" \\
  -d '{"horizons": [24, 48]}'</div>
            </div>

            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/model/info</div>
                <p>Get detailed model information and performance metrics</p>
                <div class="example">curl -X GET "{{ base_url }}/model/info"</div>
            </div>
        </div>

        <!-- Usage Examples -->
        <div class="section">
            <h2>💡 Usage Examples</h2>

            <h3>Python Example</h3>
            <div class="example">import requests
import json

# Make a forecast request
response = requests.post('{{ base_url }}/forecast',
                        json={'hours_ahead': 48})

if response.status_code == 200:
    forecast = response.json()
    print(f"48-hour forecast: {forecast['predictions'][:5]}...")  # First 5 values
else:
    print(f"Error: {response.status_code}")</div>

            <h3>JavaScript Example</h3>
            <div class="example">fetch('{{ base_url }}/forecast', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        hours_ahead: 24
    })
})
.then(response => response.json())
.then(data => {
    console.log('24-hour forecast:', data.predictions);
})
.catch(error => console.error('Error:', error));</div>
        </div>

        <div class="footer">
            <p>🔬 <strong>GridCast</strong> - Intelligent Energy Demand Forecasting System</p>
            <p>Built with Flask, scikit-learn, PyTorch, and MLflow</p>
        </div>
    </div>
</body>
</html>
"""

def load_model():
    """Load the champion model on startup"""
    global predictor, model_info

    try:
        # Path to champion model
        model_path = Path(__file__).parent.parent / "models" / "saved_models" / "champion_model.joblib"

        if not model_path.exists():
            logger.warning(f"Champion model not found at {model_path}")
            return False

        if GridCastPredictor is None:
            logger.error("GridCastPredictor class not available")
            return False

        # Load model (assuming XGBoost for demo)
        predictor = GridCastPredictor(str(model_path), "xgboost")

        if predictor.is_loaded:
            model_info = predictor.get_model_info()
            logger.info(f"Model loaded successfully: {model_info['model_type']}")
            return True
        else:
            logger.error("Failed to load model")
            return False

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_sample_data(hours=168):
    """Generate sample energy demand data for demo purposes"""
    # Create sample data (1 week by default)
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=hours)

    dates = pd.date_range(start=start_time, end=end_time, freq='H')[:-1]  # Exclude end point

    # Generate realistic demand pattern
    hours_array = np.arange(len(dates))
    time_of_day = dates.hour
    day_of_week = dates.dayofweek

    # Base demand with daily/weekly patterns
    base_demand = 12000
    daily_pattern = 3000 * np.sin(2 * np.pi * (time_of_day - 6) / 24)  # Peak around 6 PM
    weekly_pattern = 1000 * (1 - 0.3 * (day_of_week >= 5))  # Lower on weekends
    seasonal = 500 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    noise = np.random.normal(0, 300, len(dates))

    demand = base_demand + daily_pattern + weekly_pattern + seasonal + noise
    demand = np.maximum(demand, 6000)  # Minimum demand

    return pd.DataFrame({'demand_mw': demand}, index=dates)

@app.route('/')
def home():
    """Serve the API documentation page"""
    try:
        # Determine API status
        if predictor and predictor.is_loaded:
            api_status = "🟢 Online and Ready"
            model_status = f"✅ {model_info['model_type'].upper()} Model Loaded"
            status_class = "healthy"
        else:
            api_status = "🟡 Online (Model Issues)"
            model_status = "❌ Model Not Available"
            status_class = "error"

        base_url = request.url_root.rstrip('/')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        return render_template_string(
            HTML_TEMPLATE,
            api_status=api_status,
            model_status=model_status,
            status_class=status_class,
            timestamp=timestamp,
            base_url=base_url,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy' if (predictor and predictor.is_loaded) else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': predictor is not None and predictor.is_loaded,
            'model_type': model_info['model_type'] if model_info else None,
            'api_version': '1.0.0',
            'forecast_capability': {
                'max_hours': 48,
                'available': predictor is not None and predictor.is_loaded
            }
        }

        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate energy demand forecast"""
    try:
        # Check if model is available
        if not predictor or not predictor.is_loaded:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 503

        # Parse request
        data = request.get_json() if request.is_json else {}
        hours_ahead = data.get('hours_ahead', 48)

        # Validate parameters
        if not isinstance(hours_ahead, int) or hours_ahead < 1 or hours_ahead > 48:
            return jsonify({
                'error': 'hours_ahead must be an integer between 1 and 48',
                'success': False
            }), 400

        # Get historical data (use provided data or generate sample)
        if 'data' in data and data['data']:
            # User provided historical data
            try:
                hist_data = pd.DataFrame(data['data'])
                if 'timestamp' in hist_data.columns:
                    hist_data['timestamp'] = pd.to_datetime(hist_data['timestamp'])
                    hist_data.set_index('timestamp', inplace=True)
            except Exception as e:
                return jsonify({
                    'error': f'Invalid data format: {str(e)}',
                    'success': False
                }), 400
        else:
            # Generate sample data
            hist_data = generate_sample_data(hours=168)  # 1 week

        # Make prediction
        result = predictor.predict(hist_data, hours_ahead=hours_ahead)

        if result['success']:
            # Add metadata
            result['metadata'] = {
                'request_time': datetime.now().isoformat(),
                'model_type': model_info['model_type'],
                'data_points_used': len(hist_data),
                'last_data_point': hist_data.index[-1].isoformat() if hasattr(hist_data.index[-1], 'isoformat') else str(hist_data.index[-1])
            }

            logger.info(f"Forecast generated: {hours_ahead} hours, {len(result['predictions'])} predictions")
            return jsonify(result)
        else:
            logger.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Forecast endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'success': False,
            'details': str(e)
        }), 500

@app.route('/forecast/batch', methods=['POST'])
def batch_forecast():
    """Generate multiple forecasts with different horizons"""
    try:
        if not predictor or not predictor.is_loaded:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 503

        data = request.get_json() if request.is_json else {}
        horizons = data.get('horizons', [24, 48])

        # Validate horizons
        if not isinstance(horizons, list) or not all(isinstance(h, int) and 1 <= h <= 48 for h in horizons):
            return jsonify({
                'error': 'horizons must be a list of integers between 1 and 48',
                'success': False
            }), 400

        # Generate sample data
        hist_data = generate_sample_data(hours=168)

        # Generate forecasts for each horizon
        results = {}
        for horizon in horizons:
            result = predictor.predict(hist_data, hours_ahead=horizon)
            if result['success']:
                results[f'{horizon}h'] = {
                    'predictions': result['predictions'],
                    'timestamps': result['timestamps'],
                    'horizon_hours': horizon
                }
            else:
                results[f'{horizon}h'] = {
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                }

        return jsonify({
            'success': True,
            'forecasts': results,
            'metadata': {
                'request_time': datetime.now().isoformat(),
                'model_type': model_info['model_type'],
                'horizons_requested': horizons
            }
        })

    except Exception as e:
        logger.error(f"Batch forecast error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'success': False,
            'details': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info_endpoint():
    """Get detailed model information"""
    try:
        if not predictor:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 503

        # Load model metadata if available
        metadata_path = Path(__file__).parent.parent / "models" / "saved_models" / "champion_model_metadata.json"
        metadata = {}

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model metadata: {e}")

        info = {
            'success': True,
            'model_info': model_info,
            'performance_metrics': metadata.get('performance_metrics', {}),
            'model_description': metadata.get('model_description', 'Champion model for energy demand forecasting'),
            'capabilities': {
                'max_forecast_horizon': 48,
                'supported_formats': ['json'],
                'real_time_inference': True,
                'batch_processing': True
            },
            'last_updated': metadata.get('packaging_timestamp', 'Unknown')
        }

        return jsonify(info)

    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'success': False,
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False,
        'available_endpoints': [
            'GET /',
            'GET /health',
            'POST /forecast',
            'POST /forecast/batch',
            'GET /model/info'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("🚀 Starting GridCast API Server...")
    print("=================================")

    # Load model on startup
    model_loaded = load_model()

    if model_loaded:
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model not available - API will run in demo mode")

    print("\\n📡 API Endpoints:")
    print("   GET  /          - API documentation")
    print("   GET  /health    - Health check")
    print("   POST /forecast  - Generate forecast")
    print("   POST /forecast/batch - Batch forecasts")
    print("   GET  /model/info - Model information")

    print("\\n🌐 Starting server on http://localhost:5001")
    print("   Press Ctrl+C to stop the server")

    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)