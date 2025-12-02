# sse.py - Server-Sent Events handling

import queue
import threading
import json
from typing import Dict, Any

# SSE clients
sse_clients = []
sse_clients_lock = threading.Lock()


def broadcast_sse_event(event_data: Dict[str, Any], app_logger=None):
    """Gửi sự kiện đến tất cả SSE clients"""
    with sse_clients_lock:
        dead_clients = []
        for client_queue in sse_clients:
            try:
                client_queue.put_nowait(event_data)
            except queue.Full:
                # Client queue is full, mark for removal
                dead_clients.append(client_queue)
            except Exception as e:
                if app_logger:
                    app_logger.error(f"Error broadcasting SSE event: {e}")
                dead_clients.append(client_queue)

        # Remove dead clients
        for dead_client in dead_clients:
            try:
                sse_clients.remove(dead_client)
            except ValueError:
                pass


def add_sse_client(client_queue: queue.Queue):
    """Thêm SSE client mới"""
    with sse_clients_lock:
        sse_clients.append(client_queue)


def remove_sse_client(client_queue: queue.Queue):
    """Xóa SSE client"""
    with sse_clients_lock:
        try:
            sse_clients.remove(client_queue)
        except ValueError:
            pass


def generate_sse_stream(client_queue: queue.Queue, app_logger=None):
    """Generator cho SSE stream"""
    add_sse_client(client_queue)
    try:
        while True:
            try:
                # Wait for event with timeout
                event_data = client_queue.get(timeout=30)
                # Format as SSE
                event_type = event_data.get('type', 'message')
                event_json = json.dumps(event_data)
                yield f"event: {event_type}\ndata: {event_json}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield ": heartbeat\n\n"
    except GeneratorExit:
        # Client disconnected
        remove_sse_client(client_queue)
    except Exception as e:
        if app_logger:
            app_logger.error(f"SSE stream error: {e}")
        remove_sse_client(client_queue)