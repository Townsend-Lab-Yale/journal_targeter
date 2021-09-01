import logging
from datetime import datetime

from . import db

_logger = logging.getLogger(__name__)


class Source(db.Model):
    __tablename__ = 'source'
    source_name = db.Column(db.String(255), primary_key=True)
    update_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Source {self.source_name}>'

    @staticmethod
    def updated_now(source_name):
        s = Source.query.filter_by(source_name=source_name).first()
        now = datetime.utcnow()
        if s is None:
            s = Source(source_name=source_name, update_time=now)
        else:
            s.update_time = now
        db.session.add(s)
        db.session.commit()
        _logger.info(f"New update time stored for {source_name}.")

    @staticmethod
    def store_update(source_name=None, update_time=None):
        s = Source.query.filter_by(source_name=source_name).first()
        if s is not None:
            s.update_time = update_time
        else:
            s = Source(source_name=source_name, update_time=update_time)
        _logger.info(f"New update time stored for {source_name}.")
        db.session.add(s)
        db.session.commit()
