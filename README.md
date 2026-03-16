# Pothole Detection System (Zimbabwe)

This system helps in detecting and reporting potholes in Zimbabwe using computer vision and mapping.

## Getting Started

You can start both the backend and frontend at the same time using the `main.py` script.

### Prerequisites

- Python 3.x
- Flask and other dependencies (see `backend/requirements.txt`)

### Installation

1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

### Running the System

To start both the backend API and the frontend dashboard:

```bash
python main.py
```

This will:
1. Start the Flask backend on [http://localhost:5000](http://localhost:5000)
2. Start a local server for the frontend on [http://localhost:8000](http://localhost:8000)
3. Automatically open your default web browser to the application.

### Project Structure

- `backend/`: Flask application for API and pothole detection.
- `frontend/`: HTML/JS/CSS files for the user dashboard.
- `database/`: SQL schema for the database.
- `main.py`: Orchestration script to run everything together.
