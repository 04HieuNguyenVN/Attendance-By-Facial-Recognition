"""
API routes for Server-Sent Events (SSE)
Các API endpoint cho real-time events
"""
from flask import Blueprint, Response
import json
import queue
from app.utils.session_utils import (
    serialize_session_payload,
    get_active_attendance_session
)
from app import globals as app_globals

events_api_bp = Blueprint('events_api', __name__, url_prefix='/api/events')


@events_api_bp.route('/stream')
def api_events_stream():
    """Server-Sent Events stream cho thông báo real-time"""
    
    def event_stream():
        # Tạo queue cho client này
        client_queue = queue.Queue(maxsize=10)
        
        # Thêm vào danh sách clients
        with app_globals.sse_clients_lock:
            app_globals.sse_clients.append(client_queue)

        initial_session = serialize_session_payload(get_active_attendance_session())
        if initial_session:
            try:
                client_queue.put_nowait({'type': 'session_updated', 'data': initial_session})
            except queue.Full:
                pass
        
        try:
            # Gửi event kết nối thành công
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            
            # Lắng nghe events từ queue
            while True:
                try:
                    event_data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except queue.Empty:
                    # Gửi heartbeat để giữ kết nối
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            # Client đã ngắt kết nối
            with app_globals.sse_clients_lock:
                if client_queue in app_globals.sse_clients:
                    app_globals.sse_clients.remove(client_queue)
    
    return Response(event_stream(), mimetype='text/event-stream')
