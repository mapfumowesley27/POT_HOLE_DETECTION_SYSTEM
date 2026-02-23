from flask import Blueprint, render_template, send_from_directory
import os

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return {
        'message': 'Pothole Detection System API',
        'status': 'running',
        'endpoints': {
            'report': '/api/report (POST)',
            'potholes': '/api/potholes (GET)',
            'alerts': '/api/alerts (GET)',
            'ml': '/ml/* (Machine Learning endpoints)'
        }
    }

@bp.route('/health')
def health():
    return {'status': 'healthy'}, 200

@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    return send_from_directory(uploads_dir, filename)