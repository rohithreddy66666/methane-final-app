# routes.py
from flask import render_template, jsonify, send_from_directory, request, url_for, send_file
from methane_detection import process_emit_data
import time
import traceback
import logging
import os
import ee
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from nixtla import NixtlaClient
import plotly
import plotly.graph_objs as go
import json

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Nixtla client with environment variable
from dotenv import load_dotenv
load_dotenv()
nixtla_client = NixtlaClient(api_key=os.getenv('NIXTLA_API_KEY'))

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def save_analysis_results_to_json(results, output_dir='templates/assets/results'):
    """Save analysis results to a JSON file"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'analysis_results_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)
        
        serializable_results = {
            'timestamp': timestamp,
            'analysis_date': datetime.now().isoformat(),
            'results': {
                'lstm_forecast': [float(x) for x in results['lstm_forecast']],
                'nixtla_forecast': [float(x) for x in results['nixtla_forecast']],
                'dates': results['dates'],
                'monthly_means': {
                    'lstm': float(results['monthly_means']['lstm']),
                    'nixtla': float(results['monthly_means']['nixtla'])
                },
                'metrics': {
                    'lstm': {
                        'rmse': float(results['lstm_metrics']['rmse']),
                        'mae': float(results['lstm_metrics']['mae'])
                    },
                    'nixtla': {
                        'rmse': float(results['nixtla_metrics']['rmse']),
                        'mae': float(results['nixtla_metrics']['mae'])
                    }
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        logger.info(f"Analysis results saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving results to JSON: {str(e)}")
        raise

def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/methane')
    def methane():
        return render_template('methane.html')

    @app.route('/time')
    def time_series():
        return render_template('time.html')

    @app.route('/get_methane_data', methods=['POST'])
    def get_methane_data():
        try:
            data = request.json
            geometry = ee.Geometry.Rectangle(data['bounds'])
            start_date = data['start_date']
            end_date = data['end_date']
            
            logger.info(f"Fetching methane data for area: {data['bounds']}, from {start_date} to {end_date}")
            
            # Load the Sentinel-5P TROPOMI CH4 dataset
            s5p = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4') \
                .select('CH4_column_volume_mixing_ratio_dry_air') \
                .filterDate(start_date, end_date)
            
            def calculate_mean(image):
                mean = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=1000,
                    maxPixels=1e9
                )
                return image.set('mean', mean.get('CH4_column_volume_mixing_ratio_dry_air'))

            s5p_with_means = s5p.map(calculate_mean)
            
            time_series = s5p_with_means.aggregate_array('mean').getInfo()
            dates = s5p_with_means.aggregate_array('system:time_start').getInfo()
            
            valid_data = [(d, v) for d, v in zip(dates, time_series) if v is not None]
            
            if not valid_data:
                logger.warning("No valid data points found")
                return jsonify({
                    'status': 'error', 
                    'message': 'No valid data points found for the selected region and time period'
                })
            
            valid_dates, valid_values = zip(*valid_data)
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in valid_dates],
                'CH4': valid_values
            })
            
            csv_path = os.path.join(app.root_path, 'methane_data.csv')
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Methane data saved to {csv_path}")
            return jsonify({
                'status': 'success', 
                'message': 'Data saved to CSV', 
                'csv_path': csv_path
            })
        
        except Exception as e:
            logger.error(f"Error in get_methane_data: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error', 
                'message': f'An error occurred: {str(e)}'
            })

    @app.route('/analyze_data', methods=['POST'])
    def analyze_data():
        csv_path = os.path.join(app.root_path, 'methane_data.csv')
        if not os.path.exists(csv_path):
            return jsonify({
                'status': 'error', 
                'message': 'CSV file not found. Please fetch data first.'
            })

        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            latest_date = df['date'].max()
            forecast_dates = pd.date_range(
                start=latest_date + pd.Timedelta(days=1),
                periods=30,
                freq='D'
            )
            
            # LSTM Analysis
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df['CH4'].values.reshape(-1, 1))
            
            seq_length = 10
            X, y = create_sequences(scaled_data, seq_length)
            
            train_size = int(len(X) * 0.8)
            X_train = torch.FloatTensor(X[:train_size])
            y_train = torch.FloatTensor(y[:train_size])
            X_test = torch.FloatTensor(X[train_size:])
            y_test = torch.FloatTensor(y[train_size:])
            
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                last_sequence = torch.FloatTensor(scaled_data[-seq_length:]).reshape(1, seq_length, 1)
                lstm_predictions = []
                
                current_sequence = last_sequence.clone()
                for _ in range(30):
                    pred = model(current_sequence)
                    lstm_predictions.append(pred.item())
                    
                    new_sequence = current_sequence.squeeze(0)[:, 0][1:]
                    new_value = pred.reshape(1)
                    current_sequence = torch.cat([new_sequence, new_value]).reshape(1, seq_length, 1)
            
            lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
            
            # Calculate LSTM metrics
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test).numpy()
                y_test_numpy = y_test.numpy()
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                lstm_rmse = np.sqrt(mean_squared_error(y_test_numpy, test_predictions))
                lstm_mae = mean_absolute_error(y_test_numpy, test_predictions)
            
            # Nixtla analysis
            df_nixtla = df.rename(columns={'date': 'ds', 'CH4': 'y'})
            fcst_df = nixtla_client.forecast(df_nixtla, h=30, level=[80, 90], freq='D')
            
            nixtla_rmse = np.sqrt(mean_squared_error(df_nixtla['y'].tail(5), fcst_df['TimeGPT'][:5]))
            nixtla_mae = mean_absolute_error(df_nixtla['y'].tail(5), fcst_df['TimeGPT'][:5])
            
            lstm_mean = np.mean(lstm_predictions)
            nixtla_mean = np.mean(fcst_df['TimeGPT'])
            
            # Create visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['date'].dt.strftime('%Y-%m-%d'),
                y=df['CH4'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates.strftime('%Y-%m-%d'),
                y=lstm_predictions.flatten(),
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates.strftime('%Y-%m-%d'),
                y=fcst_df['TimeGPT'],
                mode='lines+markers',
                name='Nixtla Forecast',
                line=dict(color='red')
            ))
            
            fig.add_hline(y=lstm_mean, 
                         line_dash="dash", 
                         line_color="green", 
                         annotation_text=f"LSTM Mean: {lstm_mean:.2f}")
            fig.add_hline(y=nixtla_mean, 
                         line_dash="dash", 
                         line_color="red", 
                         annotation_text=f"Nixtla Mean: {nixtla_mean:.2f}")
            
            fig.update_layout(
                title='CH4 Time Series Forecast - 30 Days Prediction',
                xaxis_title='Date',
                yaxis_title='CH4 Concentration (ppb)',
                legend=dict(y=0.99, x=0.01, yanchor='top', xanchor='left'),
                hovermode='x unified',
                showlegend=True,
                height=600
            )
            
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            results = {
                'lstm_forecast': lstm_predictions.flatten().tolist(),
                'nixtla_forecast': fcst_df['TimeGPT'].tolist(),
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'latest_historical_date': latest_date.strftime('%Y-%m-%d'),
                'monthly_means': {
                    'lstm': float(lstm_mean),
                    'nixtla': float(nixtla_mean)
                },
                'lstm_metrics': {
                    'rmse': float(lstm_rmse),
                    'mae': float(lstm_mae)
                },
                'nixtla_metrics': {
                    'rmse': float(nixtla_rmse),
                    'mae': float(nixtla_mae)
                },
                'plot_json': plot_json
            }
            
            # Save results to JSON
            json_path = save_analysis_results_to_json(results)
            results['json_path'] = json_path
            
            return jsonify(results)

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error', 
                'message': f'Error during analysis: {str(e)}'
            })

    @app.route('/download_csv')
    def download_csv():
        csv_path = os.path.join(app.root_path, 'methane_data.csv')
        if os.path.exists(csv_path):
            return send_file(
                csv_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name='methane_data.csv'
            )
        else:
            return jsonify({
                'status': 'error',
                'message': 'CSV file not found'
            }), 404

    @app.route('/get_analysis_json/<filename>')
    def get_analysis_json(filename):
        """Serve analysis JSON files"""
        try:
            return send_from_directory('templates/assets/results', filename)
        except Exception as e:
            logger.error(f"Error serving JSON file {filename}: {str(e)}")
            return jsonify({
                'error': f'File not found: {str(e)}'
            }), 404

    @app.route('/process', methods=['POST'])
    def process():
        try:
            # Get EMIT ID from request
            emit_id = request.json.get('emit_id')
            if not emit_id:
                return jsonify({
                    'success': False,
                    'error': 'EMIT ID is required'
                }), 400

            logger.info(f"Starting processing for EMIT ID: {emit_id}")
            start_time = time.time()
            
            # Process the data
            try:
                results = process_emit_data(emit_id, output_dir='templates/assets/results')
                
                # Update visualization path
                if results.get('visualization_path'):
                    filename = os.path.basename(results['visualization_path'])
                    results['visualization_path'] = url_for('static', 
                        filename=f'results/{filename}')
                    logger.info(f"Image path set to: {results['visualization_path']}")
                
            except Exception as e:
                logger.error(f"Error in process_emit_data: {str(e)}")
                raise
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return jsonify({
                'success': True,
                'results': results,
                'processing_time': processing_time,
                'message': 'Processing completed successfully'
            })

        except Exception as e:
            logger.error("Error occurred during processing:")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/results/<path:filename>')
    def get_result(filename):
        """Serve result files"""
        try:
            return send_from_directory('templates/assets/results', filename)
        except Exception as e:
            logger.error(f"Error serving file {filename}: {str(e)}")
            return jsonify({'error': f'File not found: {str(e)}'}), 404

    @app.route('/status')
    def status():
        """Health check endpoint"""
        return jsonify({
            'status': 'running',
            'message': 'API is operational'
        })

    @app.route('/debug/results')
    def debug_results():
        """Debug endpoint to check results directory"""
        results_dir = os.path.join('templates', 'assets', 'results')
        try:
            files = os.listdir(results_dir)
            return jsonify({
                'results_dir': results_dir,
                'files': files,
                'static_folder': app.static_folder,
                'static_url_path': app.static_url_path,
                'exists': os.path.exists(results_dir),
                'is_dir': os.path.isdir(results_dir),
                'permissions': oct(os.stat(results_dir).st_mode)[-3:],
                'absolute_path': os.path.abspath(results_dir)
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'results_dir': results_dir,
                'static_folder': app.static_folder,
                'static_url_path': app.static_url_path
            })

    @app.route('/debug/directories')
    def debug_directories():
        """Debug endpoint to check all relevant directories"""
        directories = {
            'static': app.static_folder,
            'templates': app.template_folder,
            'uploads': 'uploads',
            'results': 'templates/assets/results'
        }
        
        status = {}
        for name, path in directories.items():
            status[name] = {
                'path': path,
                'exists': os.path.exists(path),
                'is_directory': os.path.isdir(path) if os.path.exists(path) else False,
                'absolute_path': os.path.abspath(path),
                'contents': os.listdir(path) if os.path.exists(path) and os.path.isdir(path) else None
            }
        
        return jsonify(status)

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    return app