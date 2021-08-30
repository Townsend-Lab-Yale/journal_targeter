from datetime import datetime

from . import db


class Source(db.Model):
    __tablename__ = 'source'
    source_name = db.Column(db.String(255), primary_key=True)
    update_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Source {self.source_name}>'
