import logging
from datetime import datetime
from importlib import resources

import yaml
from flask import Flask

from . import db

_logger = logging.getLogger(__name__)
REFS_PKG = 'journal_targeter.refs'


class Source(db.Model):
    __tablename__ = 'source'
    source_name = db.Column(db.String(255), primary_key=True)
    update_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Source {self.source_name}>'

    @staticmethod
    def updated_now(source_name):
        now = datetime.utcnow()
        Source.updated_at(source_name, now)

    @staticmethod
    def updated_at(source_name=None, update_time=None) -> None:
        s = Source.query.filter_by(source_name=source_name).first()
        prev_time = s.update_time if s is not None else None
        if prev_time is None:
            s = Source(source_name=source_name, update_time=update_time)
        elif prev_time >= update_time:
            return  # No update necessary
        else:
            s.update_time = update_time
        _logger.info(f"New update time stored for {source_name}.")
        db.session.add(s)
        db.session.commit()


class SourceTracker:

    def __init__(self):
        self.dates_repo = dict()
        self.dates_user = dict()
        # self.source_dates = None

    def init_app(self, app: Flask):
        self.dates_repo.update(SourceTracker._get_source_dates_repo())
        self.dates_user.update(SourceTracker._get_source_dates_user(app))

    def is_repo_newer_or_not_in_local_db(self, source) -> bool:
        if source in self.dates_user and \
                self.dates_user[source] >= self.dates_repo[source]:
            return False
        return True

    def refresh_dates_user(self, app) -> None:
        """Run this after manually changing/overwriting user files."""
        self.dates_user = self._get_source_dates_user(app)

    @staticmethod
    def _get_source_dates_repo():
        with resources.path(REFS_PKG, 'updates.yaml') as path:
            with open(path, 'r') as infile:
                dates_repo = yaml.load(infile, yaml.SafeLoader)

        for source, date_str in dates_repo.items():
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            dates_repo[source] = dt
        return dates_repo

    @staticmethod
    def _get_source_dates_user(app):
        with app.app_context():
            recs = Source.query.all()
        dates_user = dict()
        for r in recs:
            dates_user[r.source_name] = r.update_time
        return dates_user
