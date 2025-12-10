"""
Middleware package
"""
from .auth import register_auth_middleware

__all__ = ['register_auth_middleware']
