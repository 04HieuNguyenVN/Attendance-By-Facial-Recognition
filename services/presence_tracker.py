"""Presence tracker helper.

Tracks last-seen timestamps and accumulated presence time per student for a session.
"""
from datetime import datetime, timedelta
from pathlib import Path
import csv


class PresenceTracker:
    def __init__(self):
        # name -> {'last_seen': datetime, 'total_seconds': int}
        self._data = {}
        self._session_start = datetime.now()

    def update(self, name: str, seen_time: datetime = None):
        if seen_time is None:
            seen_time = datetime.now()
        rec = self._data.get(name)
        if rec is None:
            self._data[name] = {'last_seen': seen_time, 'total_seconds': 0}
            return
        # compute delta since last seen, but only add if gap is small (continuous presence)
        last = rec['last_seen']
        delta = (seen_time - last).total_seconds()
        if delta < 120:
            # consider continuous presence; add delta
            rec['total_seconds'] += int(delta)
        # update last seen
        rec['last_seen'] = seen_time

    def get_summary(self):
        out = []
        for name, rec in self._data.items():
            minutes = rec['total_seconds'] // 60
            out.append({'name': name, 'last_seen': rec['last_seen'], 'minutes': minutes})
        return out

    def save_csv(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'last_seen', 'total_minutes'])
            for r in self.get_summary():
                writer.writerow([r['name'], r['last_seen'].isoformat(), r['minutes']])

    def session_filename(self):
        ts = self._session_start.strftime('%Y%m%d_%H%M%S')
        return Path('attendance_sessions') / f'session_{ts}.csv'
