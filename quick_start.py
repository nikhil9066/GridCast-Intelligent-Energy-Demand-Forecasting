#!/usr/bin/env python3
"""
GridCast Quick Start Script
Run this to see a minimal working version of GridCast
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_demo_data():
    """Generate demo energy consumption data"""
    print("📊 Generating demo energy data...")

    # Create 7 days of hourly data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='h'
    )[:-1]

    # Realistic energy pattern
    base_demand = 12000
    daily_pattern = 3000 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
    weekly_pattern = 1000 * (1 - 0.3 * (dates.dayofweek >= 5))
    noise = np.random.normal(0, 300, len(dates))

    demand = base_demand + daily_pattern + weekly_pattern + noise

    return pd.DataFrame({'demand_mw': demand}, index=dates)

def simple_forecast(data, hours_ahead=48):
    """Simple forecasting using seasonal naive method"""
    print(f"🔮 Generating {hours_ahead}-hour forecast...")

    # Use same hour from previous week as forecast
    last_week_data = data.tail(24 * 7)  # Last week

    # Forecast next 48 hours using seasonal pattern
    forecast = []
    for i in range(hours_ahead):
        # Use value from same hour last week
        historical_idx = i % len(last_week_data)
        forecast_value = last_week_data.iloc[historical_idx]['demand_mw']
        # Add some trend and noise
        trend = np.random.normal(0, 100)
        forecast.append(forecast_value + trend)

    # Create forecast timestamps
    start_time = data.index[-1] + timedelta(hours=1)
    forecast_times = pd.date_range(start=start_time, periods=hours_ahead, freq='h')

    return {
        'success': True,
        'predictions': forecast,
        'timestamps': forecast_times.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'model_type': 'seasonal_naive',
        'forecast_horizon_hours': hours_ahead,
        'last_actual_value': float(data.iloc[-1]['demand_mw'])
    }

def create_simple_api():
    """Create a simple Flask API"""
    try:
        from flask import Flask, jsonify, render_template_string

        app = Flask(__name__)

        # Generate demo data once
        demo_data = generate_demo_data()

        @app.route('/')
        def home():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head><title>GridCast Quick Demo</title></head>
            <body style="font-family: Arial; padding: 40px; background: #f5f5f5;">
                <div style="max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                    <h1 style="color: #2c3e50; text-align: center;">⚡ GridCast Quick Demo</h1>
                    <p>This is a minimal version of GridCast running with demo data.</p>

                    <h3>🔗 Available Endpoints:</h3>
                    <ul>
                        <li><a href="/forecast">GET /forecast</a> - Generate 48h forecast</li>
                        <li><a href="/health">GET /health</a> - Check system status</li>
                        <li><a href="/data">GET /data</a> - View demo data</li>
                    </ul>

                    <h3>📊 Quick Test:</h3>
                    <button onclick="fetch('/forecast').then(r=>r.json()).then(d=>document.getElementById('result').innerHTML=JSON.stringify(d,null,2))">
                        Generate Forecast
                    </button>
                    <pre id="result" style="background: #f8f8f8; padding: 10px; margin-top: 10px;"></pre>
                </div>
            </body>
            </html>
            """)

        @app.route('/forecast')
        def forecast_endpoint():
            result = simple_forecast(demo_data)
            return jsonify(result)

        @app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'model': 'seasonal_naive',
                'data_points': len(demo_data),
                'timestamp': datetime.now().isoformat()
            })

        @app.route('/data')
        def data_endpoint():
            return jsonify({
                'data_points': len(demo_data),
                'date_range': f"{demo_data.index[0]} to {demo_data.index[-1]}",
                'mean_demand': float(demo_data['demand_mw'].mean()),
                'latest_values': demo_data.tail(5).to_dict()
            })

        return app

    except ImportError:
        print("❌ Flask not available. Install with: pip install flask")
        return None

def main():
    """Main quick start function"""
    print("🚀 GridCast Quick Start")
    print("=" * 40)

    # Generate demo data
    data = generate_demo_data()
    print(f"✅ Generated {len(data)} hours of demo data")

    # Create simple forecast
    forecast = simple_forecast(data)
    print(f"✅ Created {len(forecast['predictions'])}-hour forecast")
    print(f"   Forecast range: {min(forecast['predictions']):.0f} - {max(forecast['predictions']):.0f} MW")

    # Try to start API
    app = create_simple_api()
    if app:
        print("\n🌐 Starting simple API server...")
        print("   Visit: http://localhost:5001")
        print("   Press Ctrl+C to stop")
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("\n📊 Demo completed successfully!")
        print("   Install Flask to run the web interface: pip install flask")

if __name__ == "__main__":
    main()