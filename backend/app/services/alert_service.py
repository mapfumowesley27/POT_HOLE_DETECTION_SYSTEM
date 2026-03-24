from app.models.pothole import Alert
from app import db
import os
from datetime import datetime


class AlertService:
    def __init__(self):
        # Initialize notification services
        self.twilio_enabled = False
        self.email_enabled = False
        self.setup_services()

    def setup_services(self):
        """Setup notification services (Twilio, SendGrid)"""
        # Check if API keys are available
        if os.getenv('TWILIO_SID') and os.getenv('TWILIO_AUTH_TOKEN'):
            try:
                from twilio.rest import Client
                self.twilio_client = Client(
                    os.getenv('TWILIO_SID'),
                    os.getenv('TWILIO_AUTH_TOKEN')
                )
                self.twilio_enabled = True
            except:
                pass

        if os.getenv('SENDGRID_API_KEY'):
            try:
                import sendgrid
                self.sendgrid_client = sendgrid.SendGridAPIClient(
                    os.getenv('SENDGRID_API_KEY')
                )
                self.email_enabled = True
            except:
                pass

    def send_large_pothole_alert(self, pothole):
        """Send alert for large pothole (>1m) - Objective 3"""
        message = f"ALERT: Large pothole detected at ({pothole.latitude}, {pothole.longitude}). Diameter: {pothole.diameter:.2f}m"

        # Create alert record
        alert = Alert(
            type='large_pothole',
            pothole_id=pothole.id,
            zone_id=pothole.zone_id,
            message=message
        )
        db.session.add(alert)
        db.session.commit()

        # Send notifications
        self._send_notifications(message, alert.id, zone_id=pothole.zone_id)

        return alert

    def send_high_density_alert(self, zone, density):
        """Send alert for high density area (>5/100m²) - Objective 3"""
        message = f"ALERT: High pothole density detected in zone {zone.name}. Density: {density:.2f} potholes per 100m²"

        alert = Alert(
            type='high_density',
            zone_id=zone.id,
            message=message
        )
        db.session.add(alert)
        db.session.commit()

        self._send_notifications(message, alert.id, zone_id=zone.id)

        return alert

    def _send_notifications(self, message, alert_id, zone_id=None):
        """Send via SMS and email to respective zone authorities"""
        from app.models.user import User
        
        # Find users in this zone who are managers or admins
        query = User.query.filter(User.role.in_(['manager', 'admin']))
        if zone_id:
            query = query.filter_by(zone_id=zone_id)
        
        recipients = query.all()

        for recipient in recipients:
            # SMS via Twilio
            if self.twilio_enabled and recipient.phone_number:
                try:
                    self.twilio_client.messages.create(
                        body=message,
                        from_=os.getenv('TWILIO_PHONE'),
                        to=recipient.phone_number
                    )
                except Exception as e:
                    print(f"Twilio error for {recipient.username}: {e}")

            # Email via SendGrid
            if self.email_enabled and recipient.email:
                try:
                    from sendgrid.helpers.mail import Mail
                    mail = Mail(
                        from_email='alerts@potholedetection.com',
                        to_emails=recipient.email,
                        subject='Pothole Alert',
                        html_content=f'<p>{message}</p><p>Alert ID: {alert_id}</p>'
                    )
                    self.sendgrid_client.send(mail)
                except Exception as e:
                    print(f"SendGrid error for {recipient.username}: {e}")