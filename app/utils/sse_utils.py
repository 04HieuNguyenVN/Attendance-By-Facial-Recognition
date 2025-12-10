"""
SSE (Server-Sent Events) utilities
Các hàm tiện ích cho real-time events
"""
import queue
from app import globals as app_globals


def broadcast_sse_event(event_data):
    """Gửi sự kiện đến tất cả SSE clients"""
    with app_globals.sse_clients_lock:
        dead_clients = []
        for client_queue in app_globals.sse_clients:
            try:
                client_queue.put_nowait(event_data)
            except queue.Full:
                # Client queue is full, mark for removal
                dead_clients.append(client_queue)
            except Exception:
                dead_clients.append(client_queue)
        
        # Remove dead clients
        for dead_client in dead_clients:
            try:
                app_globals.sse_clients.remove(dead_client)
            except ValueError:
                pass
