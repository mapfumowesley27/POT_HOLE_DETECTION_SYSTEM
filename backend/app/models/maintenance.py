from app import db
from datetime import datetime


class MaintenanceCrew(db.Model):
    __tablename__ = 'maintenance_crews'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    supervisor_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    zone_id = db.Column(db.Integer, db.ForeignKey('zones.id'))
    contact_number = db.Column(db.String(20))
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    supervisor = db.relationship('User', foreign_keys=[supervisor_id], backref='supervised_crews')
    zone = db.relationship('Zone', backref='crews')
    members = db.relationship('CrewMember', backref='crew', lazy=True, cascade='all, delete-orphan')
    repair_jobs = db.relationship('RepairJob', backref='crew', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'supervisor_id': self.supervisor_id,
            'supervisor_name': self.supervisor.full_name if self.supervisor else None,
            'zone_id': self.zone_id,
            'zone_name': self.zone.name if self.zone else None,
            'contact_number': self.contact_number,
            'active': self.active,
            'member_count': len(self.members),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class CrewMember(db.Model):
    __tablename__ = 'crew_members'

    id = db.Column(db.Integer, primary_key=True)
    crew_id = db.Column(db.Integer, db.ForeignKey('maintenance_crews.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    joined_date = db.Column(db.Date, default=datetime.utcnow().date)
    active = db.Column(db.Boolean, default=True)

    # Relationships
    user = db.relationship('User', backref='crew_memberships')

    def to_dict(self):
        return {
            'id': self.id,
            'crew_id': self.crew_id,
            'crew_name': self.crew.name if self.crew else None,
            'user_id': self.user_id,
            'name': self.name,
            'role': self.role,
            'phone': self.phone,
            'joined_date': self.joined_date.isoformat() if self.joined_date else None,
            'active': self.active
        }


class Material(db.Model):
    __tablename__ = 'materials'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    unit = db.Column(db.String(20))
    quantity = db.Column(db.Float, default=0)
    reorder_level = db.Column(db.Float, default=0)
    cost_per_unit = db.Column(db.Float, default=0)
    last_restocked = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'unit': self.unit,
            'quantity': self.quantity,
            'reorder_level': self.reorder_level,
            'cost_per_unit': self.cost_per_unit,
            'low_stock': self.quantity <= self.reorder_level,
            'last_restocked': self.last_restocked.isoformat() if self.last_restocked else None
        }


class RepairJob(db.Model):
    __tablename__ = 'repair_jobs'

    id = db.Column(db.Integer, primary_key=True)
    pothole_id = db.Column(db.Integer, db.ForeignKey('potholes.id'), nullable=False)
    crew_id = db.Column(db.Integer, db.ForeignKey('maintenance_crews.id'))
    assigned_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')
    notes = db.Column(db.Text)
    before_photos = db.Column(db.JSON)
    after_photos = db.Column(db.JSON)
    quality_check_passed = db.Column(db.Boolean)
    quality_check_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    quality_check_at = db.Column(db.DateTime)

    # Relationships
    pothole = db.relationship('Pothole', backref='repair_jobs')
    assigned_by_user = db.relationship('User', foreign_keys=[assigned_by], backref='assigned_repairs')
    quality_check_user = db.relationship('User', foreign_keys=[quality_check_by], backref='quality_checks')
    materials_used = db.relationship('RepairMaterial', backref='repair_job', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'pothole_id': self.pothole_id,
            'pothole_lat': self.pothole.latitude if self.pothole else None,
            'pothole_lng': self.pothole.longitude if self.pothole else None,
            'pothole_image': self.pothole.image_path if self.pothole else None,
            'crew_id': self.crew_id,
            'crew_name': self.crew.name if self.crew else None,
            'assigned_by': self.assigned_by,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'notes': self.notes,
            'before_photos': self.before_photos or [],
            'after_photos': self.after_photos or [],
            'quality_check_passed': self.quality_check_passed,
            'repaired_by': self.quality_check_user.full_name if self.quality_check_user else None
        }


class RepairMaterial(db.Model):
    __tablename__ = 'repair_materials'

    id = db.Column(db.Integer, primary_key=True)
    repair_job_id = db.Column(db.Integer, db.ForeignKey('repair_jobs.id'), nullable=False)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), nullable=False)
    quantity_used = db.Column(db.Float, nullable=False)
    cost_per_unit = db.Column(db.Float)

    # Relationships
    material = db.relationship('Material', backref='usage_records')

    def to_dict(self):
        return {
            'id': self.id,
            'repair_job_id': self.repair_job_id,
            'material_id': self.material_id,
            'material_name': self.material.name if self.material else None,
            'quantity_used': self.quantity_used,
            'total_cost': self.quantity_used * (self.cost_per_unit or (self.material.cost_per_unit if self.material else 0))
        }