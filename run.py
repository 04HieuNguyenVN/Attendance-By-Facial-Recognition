"""
Application entry point
File khởi chạy ứng dụng Flask
"""
import sys
import io

# Thiết lập mã hóa UTF-8 cho đầu ra console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Cố gắng tải dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using default configuration.")
    print("Install it with: pip install python-dotenv")

from app import create_app

# Tạo Flask application
app = create_app()

if __name__ == '__main__':
    import os
    
    # Lấy cấu hình từ environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    
    app.logger.info(f"🚀 Starting Flask application on {host}:{port}")
    app.logger.info(f"🔧 Debug mode: {debug}")
    
    # Chạy ứng dụng
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )
