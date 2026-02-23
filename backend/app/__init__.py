from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config import Config
import os

db = SQLAlchemy()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    CORS(app)  # This will allow frontend to access the API
    db.init_app(app)

    # Create upload folder if it doesn't exist
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)

    # Register blueprints
    from app.routes import main, api, ml_routes

    app.register_blueprint(main.bp)  # No prefix for main routes
    app.register_blueprint(api.bp, url_prefix='/api')
    app.register_blueprint(ml_routes.bp, url_prefix='/ml')

    # Create database tables
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully")

    print("✅ All blueprints registered successfully")
    print(f"✅ Upload folder: {app.config.get('UPLOAD_FOLDER')}")

    return app