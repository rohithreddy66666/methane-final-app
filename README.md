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

## Deployment

### Local Development
```bash
flask run --debug
```

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t methane-analysis-app .
```

2. Run the container:
```bash
docker run -d \
  -p 5000:5000 \
  -e EARTHDATA_USERNAME=your_username \
  -e EARTHDATA_PASSWORD=your_password \
  -e OPENAI_API_KEY=your_openai_key \
  -e NIXTLA_API_KEY=your_nixtla_key \
  methane-analysis-app
```

#### Docker Environment Setup
- Ensure Docker is installed on your system
- The application uses Python 3.9 base image
- All dependencies are automatically installed during build
- Application runs on port 5000 by default
- Data persistence can be achieved by mounting volumes:
```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/templates/assets/results:/app/templates/assets/results \
  -e EARTHDATA_USERNAME=your_username \
  -e EARTHDATA_PASSWORD=your_password \
  -e OPENAI_API_KEY=your_openai_key \
  -e NIXTLA_API_KEY=your_nixtla_key \
  methane-analysis-app
```

#### Docker Compose (Optional)
Create a `docker-compose.yml` for easier deployment:
```yaml
version: '3.8'
services:
  methane-app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./templates/assets/results:/app/templates/assets/results
    environment:
      - EARTHDATA_USERNAME=${EARTHDATA_USERNAME}
      - EARTHDATA_PASSWORD=${EARTHDATA_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NIXTLA_API_KEY=${NIXTLA_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/status"]
      interval: 30s
      timeout: 30s
      retries: 3
```

Then run with:
```bash
docker-compose up -d
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
