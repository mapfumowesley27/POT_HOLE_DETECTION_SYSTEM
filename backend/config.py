import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///pothole.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML Model path
    MODEL_PATH = os.getenv('MODEL_PATH', 'app/ml/models/pothole_model.h5')

    # File upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

    # Alert thresholds (Objective 3)
    LARGE_POTHOLE_THRESHOLD = 1.0  # >1m diameter
    HIGH_DENSITY_THRESHOLD = 5  # >5 potholes per 100m²

    # API Keys
    MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')
    TWILIO_SID = os.getenv('TWILIO_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')

    # Secret key for sessions
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')