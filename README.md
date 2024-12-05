# Methane Analysis Application

## Overview
This web application provides advanced methane detection and analysis capabilities using satellite data from NASA's EMIT mission and Sentinel-5P TROPOMI. It combines real-time methane plume detection with time series analysis and forecasting to provide comprehensive insights into methane emissions and concentrations.

## Features

### Methane Plume Detection
* Process EMIT data for methane plume detection
* Utilize MAG1C algorithm for analysis
* Generate visualization of detection results
* Calculate concentration statistics

### Time Series Analysis
* Analyze historical methane concentration data from Sentinel-5P
* Generate forecasts using dual model approach:
  * LSTM (Long Short-Term Memory) neural network
  * Nixtla TimeGPT forecasting
* Interactive visualizations of trends and predictions

### Interactive Interface
* Web-based user interface
* Real-time data processing
* Interactive maps for area selection
* Downloadable analysis reports
* AI-powered chat assistant for result interpretation

## Technical Architecture

### Backend Components

#### Flask Application (app.py)
* Main application server
* Route handling
* Application configuration
* Error handling

#### API Module (api.py)
* Time series data processing
* Forecasting endpoints
* Data download functionality

#### Methane Detection (methane_detection.py)
* EMIT data processing
* MAG1C algorithm implementation
* Model caching
* Result visualization

#### Chat Bot Handler (bot_handler.py)
* OpenAI integration
* Context management
* Response processing

### Frontend Components
* Templates
  * index.html: Main landing page
  * methane.html: Methane detection interface
  * time.html: Time series analysis interface
  * chat_widget.html: Interactive chat component

### Key Technologies

#### Python Libraries
* Earth Engine API
* PyTorch
* Pandas
* NumPy
* Plotly
* Nixtla
* OpenAI

#### Frameworks
* Flask
* STARCOP (for methane detection)
* Georeader

## Setup Instructions

### Prerequisites
```bash
# Python 3.9+ required
python -m venv myenv
source myenv/bin/activate # Linux/Mac
myenv\Scripts\activate # Windows
```

### Environment Variables
Create a `.env` file with:
```
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
OPENAI_API_KEY=your_key
NIXTLA_API_KEY=your_key
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```

## API Endpoints

### Methane Detection
* **POST** `/process`
  * Process EMIT data for methane detection
  * Parameters: EMIT ID
  * Returns: Detection results and visualizations

### Time Series Analysis
* **POST** `/api/get_methane_data`
  * Fetch methane concentration data
  * Parameters: Area bounds, date range
  * Returns: Time series data
* **POST** `/api/analyze_data`
  * Generate forecasts and analysis
  * Returns: LSTM and Nixtla predictions, visualizations

### Utility Endpoints
* **GET** `/api/download_csv`
  * Download processed data
* **GET** `/api/status`
  * Service health check

## Deployment

### Local Development
```bash
flask run --debug
```

### Production Deployment (EC2)
1. Set up EC2 instance
2. Install dependencies
3. Configure environment variables
4. Set up reverse proxy (nginx recommended)
5. Use systemd for service management

## Data Sources
* NASA EMIT mission data
* Sentinel-5P TROPOMI CH4 measurements
* Earth Engine datasets

## Model Information
* STARCOP MAG1C algorithm for plume detection
* LSTM neural network for time series forecasting
* Nixtla TimeGPT for comparative forecasting

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Create pull request

## License
MIT License

## Acknowledgments
* NASA JPL for EMIT data
* ESA for Sentinel-5P data
* Google Earth Engine team
* STARCOP development team

## Support
For issues and feature requests, please use the GitHub issue tracker.
