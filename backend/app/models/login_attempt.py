from app import db
from datetime import datetime

class LoginAttempt(db.Model):
    __tablename__ = 'login_attempts'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(45))  # IPv6 can be up to 45 chars
    attempt_time = db.Column(db.DateTime, default=datetime.utcnow)
    success = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<LoginAttempt {self.username} {self.attempt_time}>'