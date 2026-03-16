from flask import Blueprint, request, jsonify, send_file
from app.models.pothole import Pothole, Alert
from app.models.user import User
from app.models.password_reset import PasswordReset
from app.ml.pothole_detector import PotholeDetector
from app import db
from datetime import datetime, timedelta
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import request, jsonify, current_app
from sqlalchemy import func
import math

import os
import io

bp = Blueprint('api', __name__)

# Constants for density calculation
DENSITY_AREA_SQM = 100  # 100 square meters
DENSITY_THRESHOLD = 5  # 5 potholes per 100m² triggers high density alert
# Approximate radius in degrees for 100m² area (at equator)
# 100m² = π * r², so r = sqrt(100/π) ≈ 5.64m
# At latitude ~17°, this is approximately 0.00005 degrees
PROXIMITY_RADIUS_DEGREES = 0.0005  # ~50m radius for clustering

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

        # Save image to disk
        if image_data:
            filename = f'pothole_{pothole.id}.jpg'
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)
            with open(upload_path, 'wb') as f:
                f.write(image_data)
            pothole.image_path = filename
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
        result = []
        for a in alerts:
            alert_dict = {
                'id': a.id,
                'type': a.type,
                'message': a.message,
                'sent_at': a.sent_at.isoformat() if a.sent_at else None,
                'acknowledged': a.acknowledged
            }
            # Add pothole info if exists
            if a.pothole_id:
                pothole = Pothole.query.get(a.pothole_id)
                if pothole:
                    alert_dict['pothole_id'] = a.pothole_id
                    alert_dict['pothole_size'] = pothole.size_classification
                    alert_dict['pothole_location'] = get_location_name(pothole.latitude, pothole.longitude)
            result.append(alert_dict)

        return jsonify(result)

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


@bp.route('/potholes/<int:pothole_id>', methods=['DELETE'])
def delete_pothole(pothole_id):
    """Delete a pothole record and its image file"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)

        # Delete image file if it exists
        if pothole.image_path:
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
            image_path = os.path.join(upload_dir, os.path.basename(pothole.image_path))
            if os.path.exists(image_path):
                os.remove(image_path)

        # Delete associated alerts
        Alert.query.filter_by(pothole_id=pothole_id).delete()

        db.session.delete(pothole)
        db.session.commit()

        return jsonify({'success': True, 'message': f'Pothole #{pothole_id} deleted successfully'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/potholes/<int:pothole_id>/annotated-image', methods=['GET'])
def get_annotated_image(pothole_id):
    """Get annotated image with detected potholes circled and labeled"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)
        if not pothole.image_path:
            return jsonify({'error': 'No image available for this pothole'}), 404

        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
        image_path = os.path.join(upload_dir, os.path.basename(pothole.image_path))
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        annotated_bytes = detector.generate_annotated_image(image_bytes)
        return send_file(
            io.BytesIO(annotated_bytes),
            mimetype='image/jpeg',
            download_name=f'annotated_{pothole.image_path}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/potholes/<int:pothole_id>/original-image', methods=['GET'])
def get_original_image(pothole_id):
    """Get the original uploaded image"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)
        if not pothole.image_path:
            return jsonify({'error': 'No image available for this pothole'}), 404

        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
        image_path = os.path.join(upload_dir, os.path.basename(pothole.image_path))
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404

        return send_file(image_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


@bp.route('/potholes/density', methods=['GET'])
def get_pothole_density():
    """
    Calculate pothole density per 100 square meters.
    Identifies potholes within the same proximity based on location (coordinates).
    Returns heat map data if density >= 5 potholes per 100m².
    """
    try:
        # Get all potholes that are not repaired
        potholes = Pothole.query.filter(Pothole.status != 'repaired').all()
        
        if not potholes:
            return jsonify({
                'total_potholes': 0,
                'high_density_areas': [],
                'heatmap_data': [],
                'message': 'No potholes found'
            })
        
        # Group potholes by proximity (clusters within ~50m radius)
        clusters = []
        processed_ids = set()
        
        for pothole in potholes:
            if pothole.id in processed_ids:
                continue
            
            # Find all potholes within proximity radius
            nearby_potholes = [pothole]
            processed_ids.add(pothole.id)
            
            for other in potholes:
                if other.id in processed_ids:
                    continue
                
                # Calculate distance using Haversine formula
                distance = calculate_distance(
                    pothole.latitude, pothole.longitude,
                    other.latitude, other.longitude
                )
                
                # If within ~50m radius (0.0005 degrees approx)
                if distance <= PROXIMITY_RADIUS_DEGREES:
                    nearby_potholes.append(other)
                    processed_ids.add(other.id)
            
            # Calculate density for this cluster
            cluster_center_lat = sum(p.latitude for p in nearby_potholes) / len(nearby_potholes)
            cluster_center_lng = sum(p.longitude for p in nearby_potholes) / len(nearby_potholes)
            
            # Estimate area in square meters (using average distance to all points)
            avg_distance = sum(
                calculate_distance(cluster_center_lat, cluster_center_lng, p.latitude, p.longitude)
                for p in nearby_potholes
            ) / max(len(nearby_potholes), 1)
            
            # Convert to approximate area (circle radius)
            area_sqm = math.pi * (avg_distance * 111000) ** 2 if avg_distance > 0 else 100
            
            # Calculate density per 100m²
            density_per_100sqm = (len(nearby_potholes) / max(area_sqm, 1)) * 100
            
            clusters.append({
                'center_latitude': cluster_center_lat,
                'center_longitude': cluster_center_lng,
                'pothole_count': len(nearby_potholes),
                'area_sqm': round(area_sqm, 2),
                'density_per_100sqm': round(density_per_100sqm, 2),
                'is_high_density': density_per_100sqm >= DENSITY_THRESHOLD,
                'pothole_ids': [p.id for p in nearby_potholes]
            })
        
        # Get high density areas (>= 5 potholes per 100m²)
        high_density_areas = [c for c in clusters if c['is_high_density']]
        
        # Generate heatmap data for all clusters
        heatmap_data = []
        for cluster in clusters:
            # Intensity based on density (0.4 for low, 0.8 for high)
            intensity = 0.4 if cluster['density_per_100sqm'] < DENSITY_THRESHOLD else 0.8
            heatmap_data.append([
                cluster['center_latitude'],
                cluster['center_longitude'],
                intensity
            ])
            
            # Create alert for high density areas
            if cluster['is_high_density'] and cluster['pothole_count'] >= 3:
                # Check if alert already exists for this area
                existing_alert = Alert.query.filter(
                    Alert.type == 'high_density',
                    Alert.sent_at >= datetime.now() - timedelta(hours=24)
                ).first()
                
                if not existing_alert:
                    alert = Alert(
                        type='high_density',
                        message=f"High pothole density detected: {cluster['pothole_count']} potholes per 100m² at location",
                        sent_at=datetime.now()
                    )
                    db.session.add(alert)
        
        db.session.commit()
        
        return jsonify({
            'total_potholes': len(potholes),
            'total_clusters': len(clusters),
            'high_density_areas': high_density_areas,
            'heatmap_data': heatmap_data,
            'density_threshold': DENSITY_THRESHOLD,
            'clusters': clusters
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula (in degrees)"""
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers * degrees to radians conversion
    # Result is in degrees
    return c / 57.2958  # Approximate degrees


# =====================================================
# NEW ADMIN/USER MANAGEMENT ROUTES
# =====================================================

@bp.route('/admin/users', methods=['GET'])
def get_users():
    """Return all users (admin only)"""
    try:
        # Check if users table exists (you need to create this)
        from app.models.user import User  # Create this model

        users = User.query.all()
        return jsonify([{
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'full_name': u.full_name,
            'role': u.role,
            'status': u.status,
            'last_active': u.last_active.isoformat() if u.last_active else None,
            'phone': u.phone_number,
            'zone_id': u.zone_id
        } for u in users])

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/admin/users', methods=['POST'])
def create_user():
    """Create new user (admin only)"""
    try:
        from app.models.user import User

        data = request.json

        # Check if username exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already taken'}), 400

        # Check if email exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400

        # Hash password (use proper hashing in production)
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()

        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash,
            full_name=data.get('full_name', ''),
            role=data.get('role', 'viewer'),
            status='active',
            phone_number=data.get('phone'),
            zone_id=data.get('zone_id')
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'User created successfully',
            'user_id': user.id
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/admin/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user details"""
    try:
        from app.models.user import User

        user = User.query.get_or_404(user_id)
        data = request.json

        if 'role' in data:
            user.role = data['role']
        if 'status' in data:
            user.status = data['status']
        if 'phone' in data:
            user.phone_number = data['phone']
        if 'zone_id' in data:
            user.zone_id = data['zone_id']

        db.session.commit()

        return jsonify({'success': True, 'message': 'User updated successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/admin/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user (admin only)"""
    try:
        from app.models.user import User

        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()

        return jsonify({'success': True, 'message': 'User deleted successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/admin/users/stats', methods=['GET'])
def get_user_stats():
    """Return user statistics"""
    try:
        from app.models.user import User

        total_users = User.query.count()
        active_managers = User.query.filter_by(role='manager', status='active').count()
        active_admins = User.query.filter_by(role='admin', status='active').count()
        pending_approvals = User.query.filter_by(status='pending').count()

        return jsonify({
            'totalUsers': total_users,
            'activeManagers': active_managers,
            'activeAdmins': active_admins,
            'pendingApprovals': pending_approvals
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/zones', methods=['GET'])
def get_zones():
    """Return all zones"""
    try:
        from app.models.zone import Zone  # Create this model

        zones = Zone.query.all()
        return jsonify([{
            'id': z.id,
            'name': z.name,
            'boundary': z.boundary  # You might want to format this
        } for z in zones])

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/potholes/monthly-stats', methods=['GET'])
def get_monthly_stats():
    """Return monthly pothole statistics"""
    try:
        current_year = datetime.now().year

        # Get counts per month for current year
        monthly_counts = db.session.query(
            extract('month', Pothole.reported_at).label('month'),
            func.count(Pothole.id).label('count')
        ).filter(
            extract('year', Pothole.reported_at) == current_year
        ).group_by(
            extract('month', Pothole.reported_at)
        ).order_by('month').all()

        # Initialize array with zeros for all months
        counts = [0] * 12
        for month, count in monthly_counts:
            counts[int(month) - 1] = count

        return jsonify({
            'year': current_year,
            'counts': counts
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400





@bp.route('/potholes/<int:pothole_id>/verify', methods=['POST'])
def verify_pothole(pothole_id):
    """Verify a pothole"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)

        data = request.json
        pothole.status = 'verified'
        pothole.verified_at = datetime.utcnow()

        # You might want to store who verified it
        if data and 'verified_by' in data:
            # Add verified_by field to your model
            pass

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Pothole verified successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/potholes/<int:pothole_id>/reject', methods=['POST'])
def reject_pothole(pothole_id):
    """Reject a pothole report"""
    try:
        pothole = Pothole.query.get_or_404(pothole_id)

        pothole.status = 'rejected'  # Add this status to your model

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Pothole report rejected'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/repairs/upload', methods=['POST'])
def upload_repair_photos():
    """Upload repair photos"""
    try:
        from app.models.repair import RepairJob

        if 'photos' not in request.files:
            return jsonify({'error': 'No photos provided'}), 400

        files = request.files.getlist('photos')
        repair_job_id = request.form.get('repair_job_id', type=int)

        uploaded_files = []
        for file in files:
            # Generate filename
            filename = f'repair_{repair_job_id}_{datetime.now().timestamp()}.jpg'
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads', 'repairs')
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)

            # Save file
            file.save(upload_path)
            uploaded_files.append(filename)

        # Update repair job with photos
        if repair_job_id:
            repair_job = RepairJob.query.get(repair_job_id)
            if repair_job:
                # Store photo paths (you might want to append instead)
                repair_job.after_photos = uploaded_files
                repair_job.status = 'completed'
                repair_job.completed_at = datetime.utcnow()
                db.session.commit()

        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} photo(s) uploaded',
            'files': uploaded_files
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# =====================================================
# AUTHENTICATION ROUTES
# =====================================================

@bp.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset"""
    try:
        from app.models.user import User
        from app.models.password_reset import PasswordReset

        data = request.get_json()
        email = data.get('email')

        user = User.query.filter_by(email=email, status='active').first()

        if not user:
            return jsonify({
                'success': False,
                'message': 'If this email exists, a reset code will be sent'
            }), 200

        # Generate reset code
        reset_code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        expires_at = datetime.now() + timedelta(minutes=5)

        # Store reset code
        password_reset = PasswordReset(
            user_id=user.id,
            code=reset_code,
            expires_at=expires_at
        )
        db.session.add(password_reset)
        db.session.commit()

        # TODO: Send email with reset code
        print(f"Reset code for {email}: {reset_code}")

        return jsonify({
            'success': True,
            'message': 'Reset code sent',
            'token': reset_code  # Remove in production
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/verify-code', methods=['POST'])
def verify_code():
    """Verify reset code"""
    try:
        from app.models.user import User
        from app.models.password_reset import PasswordReset

        data = request.get_json()
        email = data.get('email')
        code = data.get('code')

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        reset = PasswordReset.query.filter_by(
            user_id=user.id,
            code=code,
            used=False
        ).first()

        if not reset or reset.expires_at < datetime.now():
            return jsonify({
                'success': False,
                'message': 'Invalid or expired code'
            }), 400

        return jsonify({
            'success': True,
            'message': 'Code verified',
            'reset_token': code
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password"""
    try:
        from app.models.user import User
        from app.models.password_reset import PasswordReset

        data = request.get_json()
        email = data.get('email')
        code = data.get('token')
        new_password = data.get('new_password')

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        reset = PasswordReset.query.filter_by(
            user_id=user.id,
            code=code,
            used=False
        ).first()

        if not reset:
            return jsonify({'error': 'Invalid reset token'}), 400

        # Update password (use proper hashing)
        user.password_hash = hashlib.sha256(new_password.encode()).hexdigest()

        # Mark code as used
        reset.used = True

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Password reset successful'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/resend-code', methods=['POST'])
def resend_code():
    """Resend verification code"""
    try:
        from app.models.user import User
        from app.models.password_reset import PasswordReset

        data = request.get_json()
        email = data.get('email')

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Generate new code
        new_code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        expires_at = datetime.now() + timedelta(minutes=5)

        # Update existing or create new
        reset = PasswordReset.query.filter_by(user_id=user.id, used=False).first()
        if reset:
            reset.code = new_code
            reset.expires_at = expires_at
        else:
            reset = PasswordReset(
                user_id=user.id,
                code=new_code,
                expires_at=expires_at
            )
            db.session.add(reset)

        db.session.commit()

        # TODO: Send email with new code
        print(f"New reset code for {email}: {new_code}")

        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        from app.models.user import User

        data = request.get_json()

        # Check if username exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({
                'success': False,
                'message': 'Username already taken'
            }), 400

        # Check if email exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({
                'success': False,
                'message': 'Email already registered'
            }), 400

        # Hash password
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()

        # Determine status based on role
        status = 'pending' if data['role'] == 'manager' else 'active'

        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash,
            full_name=f"{data['first_name']} {data['last_name']}",
            role=data['role'],
            status=status,
            phone_number=data.get('phone'),
            zone_id=data.get('zone_id')
        )

        db.session.add(user)
        db.session.commit()

        # TODO: Send welcome email

        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'role': data['role'],
            'user_id': user.id
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/check-username', methods=['GET'])
def check_username():
    """Check if username is available"""
    try:
        from app.models.user import User

        username = request.args.get('username')

        if not username:
            return jsonify({'error': 'Username required'}), 400

        user = User.query.filter_by(username=username).first()

        return jsonify({
            'available': user is None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/auth/verify', methods=['GET'])
def verify_token():
    """Verify authentication token"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401

        token = auth_header.split(' ')[1]

        # In production, verify JWT token
        # For now, return mock user for development
        # You should implement proper JWT verification here

        from app.models.user import User

        # Mock: Find user by token (you'd decode JWT and get user_id)
        # This is just for development - replace with real JWT verification
        if token == "mock_token_for_development":
            user = User.query.first()
            if user:
                return jsonify({
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'full_name': user.full_name,
                        'email': user.email,
                        'role': user.role,
                        'status': user.status
                    }
                }), 200

        return jsonify({'error': 'Invalid token'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 401


@bp.route('/auth/login', methods=['POST', 'OPTIONS'])
def login():
    """Authenticate user and return token"""
    # Handle OPTIONS preflight request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response, 200

    try:
        from app.models.user import User
        import hashlib
        import secrets
        from datetime import datetime

        # TEMPORARILY DISABLE LOGIN ATTEMPT TRACKING
        HAS_LOGIN_ATTEMPT = False

        # Get JSON data with error handling
        data = request.get_json()

        # Debug log
        print(f"Login attempt with data: {data}")

        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        username = data.get('username')
        password = data.get('password')
        requested_role = data.get('role')

        # Validate required fields
        if not username:
            return jsonify({
                'success': False,
                'message': 'Username is required'
            }), 400

        if not password:
            return jsonify({
                'success': False,
                'message': 'Password is required'
            }), 400

        if not requested_role:
            return jsonify({
                'success': False,
                'message': 'Role is required'
            }), 400

        # Find user
        user = User.query.filter_by(username=username).first()

        if not user:
            return jsonify({
                'success': False,
                'message': 'Invalid username or password'
            }), 401

        # Check password
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if user.password_hash != password_hash:
            return jsonify({
                'success': False,
                'message': 'Invalid username or password'
            }), 401

        # Check if user is active
        if user.status != 'active':
            status_msg = {
                'pending': 'Account pending approval',
                'inactive': 'Account is inactive',
                'suspended': 'Account is suspended'
            }.get(user.status, 'Account is not active')

            return jsonify({
                'success': False,
                'message': status_msg
            }), 403

        # Check if role matches
        if user.role != requested_role:
            return jsonify({
                'success': False,
                'message': f'Invalid role. This account is registered as {user.role}.'
            }), 403

        # Update last login
        user.last_login = datetime.now()
        user.last_active = datetime.now()
        db.session.commit()

        # Generate token
        token = secrets.token_hex(32)

        response_data = {
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'full_name': user.full_name,
                'email': user.email,
                'role': user.role,
                'status': user.status
            }
        }

        # Add CORS headers
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

    except Exception as e:
        print(f"Login error: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': 'Server error. Please try again.'
        }), 500
# =====================================================
# MAINTENANCE CREW MANAGEMENT
# =====================================================

@bp.route('/crews', methods=['GET'])
def get_crews():
    """Get all maintenance crews"""
    try:
        from app.models.maintenance import MaintenanceCrew
        crews = MaintenanceCrew.query.all()
        return jsonify([c.to_dict() for c in crews])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/crews', methods=['POST'])
def create_crew():
    """Create a new maintenance crew"""
    try:
        from app.models.maintenance import MaintenanceCrew
        data = request.json
        crew = MaintenanceCrew(
            name=data['name'],
            supervisor_id=data.get('supervisor_id'),
            zone_id=data.get('zone_id'),
            contact_number=data.get('contact_number')
        )
        db.session.add(crew)
        db.session.commit()
        return jsonify(crew.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/crews/<int:crew_id>', methods=['PUT'])
def update_crew(crew_id):
    """Update a crew"""
    try:
        from app.models.maintenance import MaintenanceCrew
        crew = MaintenanceCrew.query.get_or_404(crew_id)
        data = request.json
        if 'name' in data: crew.name = data['name']
        if 'supervisor_id' in data: crew.supervisor_id = data['supervisor_id']
        if 'zone_id' in data: crew.zone_id = data['zone_id']
        if 'contact_number' in data: crew.contact_number = data['contact_number']
        if 'active' in data: crew.active = data['active']
        db.session.commit()
        return jsonify(crew.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/crews/<int:crew_id>', methods=['DELETE'])
def delete_crew(crew_id):
    """Delete a crew"""
    try:
        from app.models.maintenance import MaintenanceCrew
        crew = MaintenanceCrew.query.get_or_404(crew_id)
        db.session.delete(crew)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/crew-members', methods=['GET'])
def get_crew_members():
    """Get crew members"""
    try:
        from app.models.maintenance import CrewMember
        crew_id = request.args.get('crew_id', type=int)
        query = CrewMember.query
        if crew_id:
            query = query.filter_by(crew_id=crew_id)
        members = query.all()
        return jsonify([m.to_dict() for m in members])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/crew-members', methods=['POST'])
def create_crew_member():
    """Add a crew member"""
    try:
        from app.models.maintenance import CrewMember
        data = request.json
        member = CrewMember(
            crew_id=data['crew_id'],
            user_id=data.get('user_id'),
            name=data['name'],
            role=data.get('role'),
            phone=data.get('phone')
        )
        db.session.add(member)
        db.session.commit()
        return jsonify(member.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/crew-members/<int:member_id>', methods=['PUT'])
def update_crew_member(member_id):
    """Update a crew member"""
    try:
        from app.models.maintenance import CrewMember
        member = CrewMember.query.get_or_404(member_id)
        data = request.json
        if 'name' in data: member.name = data['name']
        if 'role' in data: member.role = data['role']
        if 'phone' in data: member.phone = data['phone']
        if 'active' in data: member.active = data['active']
        db.session.commit()
        return jsonify(member.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/crew-members/<int:member_id>', methods=['DELETE'])
def delete_crew_member(member_id):
    """Delete a crew member"""
    try:
        from app.models.maintenance import CrewMember
        member = CrewMember.query.get_or_404(member_id)
        db.session.delete(member)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/materials', methods=['GET'])
def get_materials():
    """Get all materials"""
    try:
        from app.models.maintenance import Material
        materials = Material.query.all()
        return jsonify([m.to_dict() for m in materials])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/materials', methods=['POST'])
def create_material():
    """Create a new material"""
    try:
        from app.models.maintenance import Material
        data = request.json
        material = Material(
            name=data['name'],
            unit=data.get('unit'),
            quantity=data.get('quantity', 0),
            reorder_level=data.get('reorder_level', 0),
            cost_per_unit=data.get('cost_per_unit', 0)
        )
        db.session.add(material)
        db.session.commit()
        return jsonify(material.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/materials/<int:material_id>', methods=['PUT'])
def update_material(material_id):
    """Update a material"""
    try:
        from app.models.maintenance import Material
        material = Material.query.get_or_404(material_id)
        data = request.json
        if 'name' in data: material.name = data['name']
        if 'unit' in data: material.unit = data['unit']
        if 'quantity' in data: material.quantity = data['quantity']
        if 'reorder_level' in data: material.reorder_level = data['reorder_level']
        if 'cost_per_unit' in data: material.cost_per_unit = data['cost_per_unit']
        db.session.commit()
        return jsonify(material.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/materials/<int:material_id>', methods=['DELETE'])
def delete_material(material_id):
    """Delete a material"""
    try:
        from app.models.maintenance import Material
        material = Material.query.get_or_404(material_id)
        db.session.delete(material)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

# Repair Jobs endpoints (complete)
@bp.route('/repair-jobs', methods=['GET'])
def get_repair_jobs():
    """Return repair jobs with optional status filter"""
    try:
        from app.models.maintenance import RepairJob
        status = request.args.get('status')
        query = RepairJob.query
        if status:
            query = query.filter_by(status=status)
        jobs = query.order_by(RepairJob.assigned_at.desc()).all()
        return jsonify([j.to_dict() for j in jobs])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/repair-jobs', methods=['POST'])
def create_repair_job():
    """Create a repair job"""
    try:
        from app.models.maintenance import RepairJob
        data = request.json
        job = RepairJob(
            pothole_id=data['pothole_id'],
            crew_id=data.get('crew_id'),
            assigned_by=data.get('assigned_by'),
            notes=data.get('notes')
        )
        db.session.add(job)
        db.session.commit()
        return jsonify(job.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/repair-jobs/<int:job_id>', methods=['PUT'])
def update_repair_job(job_id):
    """Update repair job (start, complete, assign crew)"""
    try:
        from app.models.maintenance import RepairJob
        job = RepairJob.query.get_or_404(job_id)
        data = request.json
        if 'status' in data:
            job.status = data['status']
            if data['status'] == 'in_progress' and not job.started_at:
                job.started_at = datetime.utcnow()
            elif data['status'] == 'completed' and not job.completed_at:
                job.completed_at = datetime.utcnow()
        if 'crew_id' in data: job.crew_id = data['crew_id']
        if 'notes' in data: job.notes = data['notes']
        if 'quality_check_passed' in data:
            job.quality_check_passed = data['quality_check_passed']
        db.session.commit()
        return jsonify(job.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/repair-jobs/<int:job_id>/photos', methods=['POST'])
def upload_repair_job_photos(job_id):
    """Upload after photos for repair job"""
    try:
        from app.models.maintenance import RepairJob
        if 'photos' not in request.files:
            return jsonify({'error': 'No photos provided'}), 400
        files = request.files.getlist('photos')
        job = RepairJob.query.get_or_404(job_id)
        uploaded_files = []
        for file in files:
            filename = f'repair_{job_id}_{datetime.now().timestamp()}.jpg'
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads', 'repairs')
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)
            file.save(upload_path)
            uploaded_files.append(filename)
        job.after_photos = uploaded_files
        job.status = 'completed'
        job.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True, 'files': uploaded_files})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@bp.route('/repair-jobs/<int:job_id>/materials', methods=['GET'])
def get_repair_job_materials(job_id):
    """Get materials used in a repair job"""
    try:
        from app.models.maintenance import RepairMaterial
        materials = RepairMaterial.query.filter_by(repair_job_id=job_id).all()
        return jsonify([m.to_dict() for m in materials])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/repair-jobs/<int:job_id>/materials', methods=['POST'])
def add_repair_job_material(job_id):
    """Add material used in repair job"""
    try:
        from app.models.maintenance import RepairMaterial, Material
        data = request.json
        material = Material.query.get(data['material_id'])
        if not material:
            return jsonify({'error': 'Material not found'}), 404
        rm = RepairMaterial(
            repair_job_id=job_id,
            material_id=data['material_id'],
            quantity_used=data['quantity_used'],
            cost_per_unit=material.cost_per_unit
        )
        # Deduct from inventory
        material.quantity -= data['quantity_used']
        db.session.add(rm)
        db.session.commit()
        return jsonify(rm.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

# Get repaired potholes for display
@bp.route('/potholes/repaired', methods=['GET'])
def get_repaired_potholes():
    """Get all repaired potholes with after photos"""
    try:
        from app.models.maintenance import RepairJob
        jobs = RepairJob.query.filter_by(status='completed').all()
        result = []
        for job in jobs:
            if job.after_photos:
                result.append({
                    'id': job.id,
                    'pothole_id': job.pothole_id,
                    'latitude': job.pothole.latitude if job.pothole else None,
                    'longitude': job.pothole.longitude if job.pothole else None,
                    'after_photos': job.after_photos,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'repaired_by': job.quality_check_user.full_name if job.quality_check_user else None
                })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =====================================================
# EMAIL FUNCTIONS
# =====================================================

def send_reset_email(to_email, code, name):
    """Send password reset email"""
    try:
        # Configure your email settings - use environment variables in production
        smtp_server = current_app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = current_app.config.get('SMTP_PORT', 587)
        sender_email = current_app.config.get('MAIL_USERNAME', 'noreply@potholezw.gov.zw')
        sender_password = current_app.config.get('MAIL_PASSWORD', '')

        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Password Reset Request - Pothole Detection Zimbabwe"
        message["From"] = sender_email
        message["To"] = to_email

        # Plain text version
        text = f"""
        Hello {name},

        You requested to reset your password. Use this verification code:

        {code}

        This code will expire in 5 minutes.

        If you didn't request this, please ignore this email.

        Regards,
        Pothole Detection Zimbabwe Team
        """

        # HTML version
        html = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: 'Inter', Arial, sans-serif; padding: 20px; }}
                    .container {{ max-width: 600px; margin: 0 auto; }}
                    .header {{ color: #1e6df2; }}
                    .code-box {{ 
                        background: linear-gradient(135deg, #f0f0f0, #ffffff);
                        padding: 20px; 
                        text-align: center; 
                        font-size: 32px; 
                        font-weight: bold; 
                        letter-spacing: 5px; 
                        border-radius: 15px;
                        border: 1px solid #ddd;
                        margin: 20px 0;
                    }}
                    .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">🔐 Password Reset Request</h2>
                    <p>Hello <strong>{name}</strong>,</p>
                    <p>You requested to reset your password. Use this verification code:</p>
                    <div class="code-box">{code}</div>
                    <p>This code will expire in <strong>5 minutes</strong>.</p>
                    <p>If you didn't request this, please ignore this email.</p>
                    <hr>
                    <div class="footer">
                        <p>Pothole Detection Zimbabwe<br>Making roads safer together</p>
                    </div>
                </div>
            </body>
        </html>
        """

        # Attach parts
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send email (uncomment when email is configured)
        # if sender_password:  # Only send if password is configured
        #     with smtplib.SMTP(smtp_server, smtp_port) as server:
        #         server.starttls()
        #         server.login(sender_email, sender_password)
        #         server.sendmail(sender_email, to_email, message.as_string())

        # For development, just print
        print(f"🔐 Reset code for {to_email}: {code}")
        print(f"To: {name} <{to_email}>")

    except Exception as e:
        print(f"Error sending email: {e}")
        # Don't fail the request if email fails in development


def send_welcome_email(to_email, name, role):
    """Send welcome email to new users"""
    try:
        smtp_server = current_app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = current_app.config.get('SMTP_PORT', 587)
        sender_email = current_app.config.get('MAIL_USERNAME', 'noreply@potholezw.gov.zw')
        sender_password = current_app.config.get('MAIL_PASSWORD', '')

        message = MIMEMultipart("alternative")
        message["Subject"] = "Welcome to Pothole Detection Zimbabwe"
        message["From"] = sender_email
        message["To"] = to_email

        # Determine message based on role
        if role == 'manager':
            welcome_text = "Your manager account is pending approval from an administrator."
            status_text = "You'll receive another email once your account is activated."
        else:
            welcome_text = "Your account is now active and ready to use."
            status_text = "You can start reporting potholes immediately."

        text = f"""
        Hello {name},

        Welcome to Pothole Detection Zimbabwe!

        {welcome_text}

        {status_text}

        What you can do:
        - Report potholes with photos
        - Track repair progress
        - View pothole maps
        - Receive alerts about road conditions

        Thank you for helping make our roads safer!

        Regards,
        Pothole Detection Zimbabwe Team
        """

        html = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: 'Inter', Arial, sans-serif; padding: 20px; }}
                    .container {{ max-width: 600px; margin: 0 auto; }}
                    .header {{ color: #1e6df2; }}
                    .button {{
                        background: linear-gradient(135deg, #1e6df2, #0a4abf);
                        color: white;
                        padding: 12px 30px;
                        text-decoration: none;
                        border-radius: 30px;
                        display: inline-block;
                        margin: 20px 0;
                    }}
                    .features {{
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 15px;
                        margin: 20px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">🚧 Welcome to Pothole Detection Zimbabwe!</h2>
                    <p>Hello <strong>{name}</strong>,</p>
                    <p>{welcome_text}</p>
                    <p><strong>{status_text}</strong></p>

                    <div class="features">
                        <h4>What you can do:</h4>
                        <ul>
                            <li>📸 Report potholes with photos</li>
                            <li>🔍 Track repair progress</li>
                            <li>🗺️ View pothole maps</li>
                            <li>🔔 Receive alerts about road conditions</li>
                        </ul>
                    </div>

                    <a href="{current_app.config.get('FRONTEND_URL', 'http://localhost:5000')}/login" class="button">
                        Go to Dashboard
                    </a>

                    <p>Thank you for helping make our roads safer!</p>

                    <hr>
                    <p style="color: #666; font-size: 12px;">
                        Pothole Detection Zimbabwe<br>
                        Making roads safer together
                    </p>
                </div>
            </body>
        </html>
        """

        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send email (uncomment when configured)
        # if sender_password:
        #     with smtplib.SMTP(smtp_server, smtp_port) as server:
        #         server.starttls()
        #         server.login(sender_email, sender_password)
        #         server.sendmail(sender_email, to_email, message.as_string())

        print(f"📧 Welcome email prepared for {to_email} (role: {role})")

    except Exception as e:
        print(f"Error sending welcome email: {e}")
