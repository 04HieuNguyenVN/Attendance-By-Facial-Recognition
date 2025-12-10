"""
Authentication routes
Xử lý đăng nhập, đăng xuất
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from database import db
from app.middleware.auth import (
    verify_user_password,
    login_user,
    logout_current_user,
    sanitize_next_url,
    PUBLIC_ENDPOINTS
)

# Thêm 'auth.login' vào PUBLIC_ENDPOINTS
PUBLIC_ENDPOINTS.add('auth.login')
PUBLIC_ENDPOINTS.add('auth.logout')

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập."""
    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    next_url = sanitize_next_url(request.form.get('next'))

    if not username or not password:
        flash('Vui lòng nhập tên đăng nhập và mật khẩu', 'danger')
        return render_template('login.html')

    user = db.get_user_by_username(username)
    if not user or not verify_user_password(user, password):
        flash('Tên đăng nhập hoặc mật khẩu không đúng', 'danger')
        return render_template('login.html')

    login_user(user)
    flash(f"Xin chào, {user.get('full_name') or username}!", 'success')
    
    return redirect(next_url or url_for('main.index'))


@auth_bp.route('/logout')
def logout():
    """Đăng xuất người dùng hiện tại."""
    logout_current_user()
    flash('Đã đăng xuất thành công', 'info')
    return redirect(url_for('auth.login'))
