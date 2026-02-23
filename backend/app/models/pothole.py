from app import db
from datetime import datetime


class Pothole(db.Model):
    __tablename__ = 'potholes'

    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(500))
    size_classification = db.Column(db.String(20))  # small, medium, large
    diameter = db.Column(db.Float)  # in meters
    confidence_score = db.Column(db.Float)
    status = db.Column(db.String(20), default='pending')  # pending, verified, repaired
    reported_by = db.Column(db.String(100))
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)
    verified_at = db.Column(db.DateTime)
    repaired_at = db.Column(db.DateTime)

    # For density calculation (Objective 2)
    zone_id = db.Column(db.Integer, db.ForeignKey('zones.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'image_path': self.image_path,
            'size_classification': self.size_classification,
            'diameter': self.diameter,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'reported_at': self.reported_at.isoformat() if self.reported_at else None
        }


class Zone(db.Model):
    __tablename__ = 'zones'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    boundary = db.Column(db.String(500))  # GeoJSON polygon
    potholes = db.relationship('Pothole', backref='zone', lazy=True)

    def calculate_density(self):
        """Calculate pothole density per 100m² (Objective 2)"""
        # This would use PostGIS for accurate area calculation
        # Simplified version for now
        active_potholes = [p for p in self.potholes if p.status != 'repaired']
        area_sqm = 10000  # This should be calculated from boundary
        density = (len(active_potholes) / area_sqm) * 100
        return density


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50))  # 'large_pothole', 'high_density'
    pothole_id = db.Column(db.Integer, db.ForeignKey('potholes.id'))
    zone_id = db.Column(db.Integer, db.ForeignKey('zones.id'))
    message = db.Column(db.String(500))
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged = db.Column(db.Boolean, default=False)