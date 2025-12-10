"""
Mô-đun quốc tế hóa (i18n) cho hệ thống điểm danh.
Hỗ trợ chuyển đổi ngôn ngữ Tiếng Việt (vi) và Tiếng Anh (en).
"""

import json
import os
from functools import wraps
from flask import Flask, session, request, g
from typing import Optional, Dict, Any

# Thư mục chứa file translation
TRANSLATIONS_DIR = os.path.join(os.path.dirname(__file__), 'translations')
SUPPORTED_LANGUAGES = ['vi', 'en']
DEFAULT_LANGUAGE = 'vi'

# Cache translations để tránh đọc file nhiều lần
_translations_cache: Dict[str, Dict[str, Any]] = {}


def load_translations(lang: str) -> Dict[str, Any]:
    """
    Tải file translation JSON cho ngôn ngữ được chỉ định.
    
    Args:
        lang: Mã ngôn ngữ ('vi' hoặc 'en')
    
    Returns:
        Dictionary chứa các chuỗi dịch
    """
    if lang in _translations_cache:
        return _translations_cache[lang]
    
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    
    file_path = os.path.join(TRANSLATIONS_DIR, f'{lang}.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
            _translations_cache[lang] = translations
            return translations
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[i18n] Error loading translations for '{lang}': {e}")
        # Fallback to empty dict
        return {}


def get_locale() -> str:
    """
    Lấy ngôn ngữ hiện tại từ session, cookie hoặc browser.
    
    Returns:
        Mã ngôn ngữ ('vi' hoặc 'en')
    """
    # 1. Kiểm tra session
    if 'language' in session:
        lang = session['language']
        if lang in SUPPORTED_LANGUAGES:
            return lang
    
    # 2. Kiểm tra cookie
    lang = request.cookies.get('language')
    if lang and lang in SUPPORTED_LANGUAGES:
        return lang
    
    # 3. Kiểm tra Accept-Language header từ browser
    best_match = request.accept_languages.best_match(SUPPORTED_LANGUAGES)
    if best_match:
        return best_match
    
    # 4. Mặc định
    return DEFAULT_LANGUAGE


def set_locale(lang: str) -> bool:
    """
    Đặt ngôn ngữ cho session hiện tại.
    
    Args:
        lang: Mã ngôn ngữ ('vi' hoặc 'en')
    
    Returns:
        True nếu thành công, False nếu ngôn ngữ không hỗ trợ
    """
    if lang not in SUPPORTED_LANGUAGES:
        return False
    
    session['language'] = lang
    return True


def translate(key: str, **kwargs) -> str:
    """
    Dịch một key sang ngôn ngữ hiện tại.
    Hỗ trợ nested keys với dấu chấm (ví dụ: 'nav.home')
    
    Args:
        key: Key cần dịch (ví dụ: 'nav.home', 'common.save')
        **kwargs: Các tham số để format chuỗi
    
    Returns:
        Chuỗi đã dịch hoặc key gốc nếu không tìm thấy
    """
    lang = getattr(g, 'current_language', None) or get_locale()
    translations = load_translations(lang)
    
    # Xử lý nested keys
    keys = key.split('.')
    value = translations
    
    try:
        for k in keys:
            value = value[k]
        
        # Format với kwargs nếu có
        if kwargs and isinstance(value, str):
            return value.format(**kwargs)
        
        return value if isinstance(value, str) else key
    except (KeyError, TypeError):
        # Trả về key gốc nếu không tìm thấy translation
        return key


def _(key: str, **kwargs) -> str:
    """
    Shorthand cho hàm translate.
    Sử dụng: {{ _('nav.home') }}
    """
    return translate(key, **kwargs)


def init_i18n(app: Flask) -> None:
    """
    Khởi tạo i18n cho Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def before_request_set_locale():
        """Đặt ngôn ngữ trước mỗi request."""
        g.current_language = get_locale()
        g.translations = load_translations(g.current_language)
    
    @app.context_processor
    def inject_i18n():
        """Inject các hàm i18n vào Jinja2 templates."""
        return {
            '_': _,
            't': translate,
            'get_locale': get_locale,
            'current_language': getattr(g, 'current_language', DEFAULT_LANGUAGE),
            'supported_languages': SUPPORTED_LANGUAGES,
            'translations': getattr(g, 'translations', {})
        }
    
    # Route để chuyển đổi ngôn ngữ
    @app.route('/set-language/<lang>')
    def set_language(lang):
        """
        Route để đổi ngôn ngữ.
        Redirect về trang trước đó sau khi đổi.
        """
        from flask import redirect, request, make_response
        
        if set_locale(lang):
            # Lấy URL trang trước đó
            referrer = request.referrer or '/'
            response = make_response(redirect(referrer))
            # Lưu vào cookie (tồn tại 1 năm)
            response.set_cookie('language', lang, max_age=365*24*60*60)
            return response
        
        return redirect('/')
    
    app.logger.info(f"[i18n] Initialized with languages: {SUPPORTED_LANGUAGES}, default: {DEFAULT_LANGUAGE}")


def reload_translations() -> None:
    """
    Reload tất cả translations từ file (xóa cache).
    Hữu ích khi cập nhật file translation mà không restart server.
    """
    global _translations_cache
    _translations_cache.clear()
    for lang in SUPPORTED_LANGUAGES:
        load_translations(lang)
