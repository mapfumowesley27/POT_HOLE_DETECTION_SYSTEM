from flask import Blueprint, request, jsonify
from app.models.pothole import Pothole, Alert
from app.ml.pothole_detector import PotholeDetector
from app import db
from datetime import datetime
import os

bp = Blueprint('api', __name__)

# Initialize detector
detector = PotholeDetector()


@bp.route('/report', methods=['POST'])
def report_pothole():
    """FR-1: Web-based Report Submission with ML detection"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            image_data = None

            # Check for base64 image
            if 'image_base64' in data:
                import base64
                base64_str = data['image_base64']
                # Remove data URL prefix if present (e.g. "data:image/png;base64,...")
                if ',' in base64_str:
                    base64_str = base64_str.split(',')[1]
                # Fix padding if needed
                missing_padding = len(base64_str) % 4
                if missing_padding:
                    base64_str += '=' * (4 - missing_padding)
                image_data = base64.b64decode(base64_str)
        else:
            data = request.form
            image_data = None

            # Check for uploaded file
            if 'image' in request.files:
                image_data = request.files['image'].read()

        # Perform ML detection if image provided
        detection_result = None
        if image_data:
            detection_result = detector.detect_pothole(image_data)

        # Create pothole record
        if detection_result and detection_result['pothole_detected']:
            # Use ML results
            size_class = detection_result['size_classification']
            diameter = detection_result['diameter']
            confidence = detection_result['confidence']
        else:
            # Use provided or default values
            size_class = data.get('size_classification', 'unknown')
            diameter = float(data.get('diameter', 0))
            confidence = float(data.get('confidence', 0))

        pothole = Pothole(
            latitude=float(data.get('latitude', 0)),
            longitude=float(data.get('longitude', 0)),
            size_classification=size_class,
            diameter=diameter,
            confidence_score=confidence,
            reported_by=data.get('reporter', 'anonymous'),
            status='pending'
        )

        db.session.add(pothole)
        db.session.commit()

        # Check if alert needed for large potholes (Objective 3)
        alert_generated = False
        if detection_result and detection_result['pothole_detected']:
            if diameter > 1.0:  # Large pothole threshold
                alert = Alert(
                    type='large_pothole',
                    pothole_id=pothole.id,
                    message=f"Large pothole detected ({diameter:.2f}m diameter) at location"
                )
                db.session.add(alert)
                db.session.commit()
                alert_generated = True

        response = {
            'success': True,
            'pothole_id': pothole.id,
            'message': 'Pothole reported successfully',
            'detection_result': detection_result,
            'alert_generated': alert_generated
        }

        return jsonify(response), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/potholes', methods=['GET'])
def get_potholes():
    """Get all potholes with optional filters"""
    try:
        status = request.args.get('status')

        query = Pothole.query
        if status:
            query = query.filter_by(status=status)

        potholes = query.all()
        return jsonify([p.to_dict() for p in potholes])

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/potholes/<int:pothole_id>', methods=['PUT'])
def update_pothole(pothole_id):
    """Update pothole status"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)
        data = request.json

        if 'status' in data:
            pothole.status = data['status']
            if data['status'] == 'verified':
                pothole.verified_at = datetime.utcnow()
            elif data['status'] == 'repaired':
                pothole.repaired_at = datetime.utcnow()

        db.session.commit()
        return jsonify(pothole.to_dict())

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get all alerts"""
    try:
        alerts = Alert.query.order_by(Alert.sent_at.desc()).all()
        return jsonify([{
            'id': a.id,
            'type': a.type,
            'message': a.message,
            'sent_at': a.sent_at.isoformat() if a.sent_at else None,
            'acknowledged': a.acknowledged
        } for a in alerts])

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        alert = Alert.query.get_or_404(alert_id)
        alert.acknowledged = True
        db.session.commit()
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    try:
        total = Pothole.query.count()
        pending = Pothole.query.filter_by(status='pending').count()
        verified = Pothole.query.filter_by(status='verified').count()
        repaired = Pothole.query.filter_by(status='repaired').count()

        return jsonify({
            'total': total,
            'pending': pending,
            'verified': verified,
            'repaired': repaired
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400