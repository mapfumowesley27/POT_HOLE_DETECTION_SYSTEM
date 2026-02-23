from flask import Blueprint, request, jsonify
from app.ml.pothole_detector import PotholeDetector
import os
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

bp = Blueprint('ml', __name__)

# Initialize detector (loads model once at startup)
detector = PotholeDetector()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/detect', methods=['POST'])
def detect_pothole():
    """
    Endpoint for pothole detection
    Accepts image file or base64 encoded image
    Returns detection results with size classification
    """
    try:
        image_data = None

        # Check if image was uploaded as file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if file and allowed_file(file.filename):
                # Read file bytes
                image_data = file.read()

                # Optionally save for reference
                filename = secure_filename(file.filename)
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                with open(upload_path, 'wb') as f:
                    f.write(image_data)

        # Check if image was sent as base64
        elif request.is_json and 'image_base64' in request.json:
            base64_str = request.json['image_base64']
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            image_data = base64.b64decode(base64_str)

        else:
            return jsonify({'error': 'No image provided'}), 400

        if image_data is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Perform detection
        result = detector.detect_pothole(image_data)

        # Add image info to response
        response = {
            'success': True,
            'detection': result,
            'message': 'Image processed successfully'
        }

        # If pothole detected, add additional info
        if result['pothole_detected']:
            response['alert'] = {
                'needs_attention': result['size_classification'] == 'large' or result['confidence'] > 0.9,
                'priority': 'high' if result['size_classification'] == 'large' else 'medium'
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/detect-batch', methods=['POST'])
def detect_batch():
    """Process multiple images at once"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        files = request.files.getlist('images')
        results = []

        for file in files:
            if file and allowed_file(file.filename):
                image_data = file.read()
                result = detector.detect_pothole(image_data)
                results.append({
                    'filename': file.filename,
                    'result': result
                })

        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_loaded': detector.model is not None,
        'model_type': 'MobileNetV2 Transfer Learning',
        'classes': detector.class_names,
        'input_size': detector.input_size,
        'status': 'ready'
    })


@bp.route('/health', methods=['GET'])
def health():
    """ML service health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })