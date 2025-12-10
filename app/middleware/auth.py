"""
Authentication middleware
Xử lý authentication, authorization và session management
"""
from flask import g, session, request, redirect, url_for, jsonify, abort
from functools import wraps
import hashlib
from werkzeug.security import check_password_hash, generate_password_hash
from database import db


# Public endpoints không cần authentication
PUBLIC_ENDPOINTS = {
    'login',
    'logout',
    'static',
}


def is_api_request():
    """Kiểm tra request hiện tại có thuộc API không."""
    path = request.path or ''
    return path.startswith('/api/')


def is_public_endpoint(endpoint):
    """Xác định endpoint có được phép truy cập công khai hay không."""
    if not endpoint:
        return False
    if endpoint == 'static' or endpoint.startswith('static.'):
        return True
    return endpoint in PUBLIC_ENDPOINTS


def sanitize_next_url(next_url):
    """Đảm bảo next_url luôn là đường dẫn nội bộ an toàn."""
    if not next_url:
        return None
    next_url = next_url.strip()
    if not next_url:
        return None
    if next_url.startswith(('http://', 'https://', '//')):
        return None
    if not next_url.startswith('/'):
        return None
    return next_url.rstrip('?') or '/'


def build_next_url():
    """Tạo giá trị next_url dựa trên request hiện tại."""
    if request.method == 'GET':
        candidate = request.full_path or request.path
    else:
        candidate = request.path
    return sanitize_next_url(candidate)


def verify_user_password(user_record, candidate_password):
    """Kiểm tra mật khẩu người dùng (hỗ trợ hash legacy)."""
    from flask import current_app as app
    
    if not user_record:
        return False
    stored_hash = user_record.get('password_hash') or ''
    if not stored_hash:
        return False

    if stored_hash.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_hash, candidate_password)

    # Legacy SHA256 hash support
    legacy_hash = hashlib.sha256(candidate_password.encode('utf-8')).hexdigest()
    if legacy_hash == stored_hash:
        try:
            new_hash = generate_password_hash(candidate_password)
            db.update_user_password(user_record['id'], new_hash)
            user_record['password_hash'] = new_hash
            app.logger.info("Đã nâng cấp hash mật khẩu cho người dùng %s", user_record.get('username'))
        except Exception as exc:
            app.logger.warning("Không thể nâng cấp hash mật khẩu: %s", exc)
        return True

    return False


def login_user(user_record):
    """Thiết lập session cho người dùng đã xác thực."""
    session.clear()
    session['user_id'] = user_record['id']
    session['user_role'] = user_record.get('role')
    session['user_name'] = user_record.get('full_name')
    session.permanent = True


def logout_current_user():
    """Đăng xuất người dùng hiện tại."""
    session.clear()


def role_required(*roles):
    """Decorator kiểm tra quyền truy cập dựa trên vai trò."""
    from flask import current_app as app
    
    allowed_roles = {role.lower() for role in roles if role}

    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                next_url = build_next_url()
                if is_api_request():
                    return jsonify({'success': False, 'message': 'Yêu cầu đăng nhập'}), 401
                if next_url:
                    return redirect(url_for('auth.login', next=next_url))
                return redirect(url_for('auth.login'))

            user_role = (user.get('role') or '').lower()
            if user_role != 'admin' and allowed_roles and user_role not in allowed_roles:
                app.logger.warning(
                    "User %s bị chặn truy cập %s (cần %s)",
                    user.get('username'),
                    request.path,
                    ','.join(allowed_roles) or 'any',
                )
                if is_api_request():
                    return jsonify({'success': False, 'message': 'Không có quyền truy cập'}), 403
                return abort(403)

            return view_func(*args, **kwargs)

        return wrapper

    return decorator


def load_logged_in_user():
    """Nạp thông tin người dùng và bảo vệ các route yêu cầu đăng nhập."""
    user_id = session.get('user_id')
    g.user = db.get_user_by_id(user_id) if user_id else None

    if is_public_endpoint(request.endpoint):
        return

    if request.path.startswith('/static/'):
        return

    if g.user is None:
        if is_api_request():
            return jsonify({'success': False, 'message': 'Yêu cầu đăng nhập'}), 401
        next_url = build_next_url()
        if next_url:
            return redirect(url_for('auth.login', next=next_url))
        return redirect(url_for('auth.login'))


def inject_user_context():
    """Cung cấp user/role hiện tại cho tất cả các template."""
    user = getattr(g, 'user', None)
    role = user.get('role') if isinstance(user, dict) else None
    return {
        'current_user': user,
        'current_role': role,
    }


def register_auth_middleware(app):
    """Đăng ký authentication middleware với Flask app."""
    app.before_request(load_logged_in_user)
    app.context_processor(inject_user_context)
