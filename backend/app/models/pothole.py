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
    boundary = db.Column(db.Text)  # Store GeoJSON polygon as text
    potholes = db.relationship('Pothole', backref='zone', lazy=True)

    def contains_point(self, lat, lon):
        """Check if a coordinate point is within this zone using basic point-in-polygon"""
        import json
        try:
            geojson = json.loads(self.boundary)
            if geojson.get('type') != 'Polygon':
                return False
            
            coords = geojson['coordinates'][0] # Outer ring
            
            # Point-in-Polygon (Ray Casting) Algorithm
            inside = False
            n = len(coords)
            p1x, p1y = coords[0]
            for i in range(n + 1):
                p2x, p2y = coords[i % n]
                if lon > min(p1x, p2x):
                    if lon <= max(p1x, p2x):
                        if lat <= max(p1y, p2y):
                            if p1x != p2x:
                                xints = (lon - p1x) * (p2y - p1y) / (p2x - p1x) + p1y
                            if p1y == p2y or lat <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        except Exception as e:
            print(f"Error checking zone boundary: {e}")
            return False

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'boundary': self.boundary
        }

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