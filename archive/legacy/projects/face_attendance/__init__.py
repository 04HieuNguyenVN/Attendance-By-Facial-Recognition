"""Face attendance package extracted from StudentAttendanceSystemByFace.

Contains AI and vision code: face detection, facenet embedding, anti-spoof,
capture and training helpers. This package is intended to be embedded into
other projects; some components still assume Django models for persistence.
"""

__all__ = [
    'reg', 'facenet', 'align', 'src', 'resources', 'Models', 'models',
    'lecturer_views', 'admin_views'
]
