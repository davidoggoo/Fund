"""
Health Dashboard for RTAI System
================================

Simple HTTP endpoint for monitoring system health.
Provides JSON health status for monitoring tools like Prometheus, Grafana, etc.
"""

import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any
import threading
import time

from rtai.utils.health import get_health_status, get_health_summary, get_health_monitor
from rtai.utils.structured_logging import configure_json_logging, get_structured_logger
from loguru import logger


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health endpoints"""
    
    def do_GET(self):
        """Handle GET requests"""
        
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query_params = parse_qs(parsed.query)
            
            if path == "/health":
                self._handle_health_check()
            elif path == "/health/status":
                self._handle_health_status()
            elif path == "/health/summary":
                self._handle_health_summary()
            elif path == "/metrics":
                self._handle_metrics()
            else:
                self._send_error(404, "Not Found")
                
        except Exception as e:
            logger.error(f"Health dashboard error: {e}")
            self._send_error(500, "Internal Server Error")
    
    def _handle_health_check(self):
        """Simple health check endpoint"""
        
        health = get_health_status()
        
        if health.status == "healthy":
            status_code = 200
        elif health.status == "degraded":
            status_code = 200  # Still OK but degraded
        else:
            status_code = 503  # Service unavailable
        
        response = {
            "status": health.status,
            "timestamp": health.timestamp,
            "uptime_seconds": health.uptime_seconds
        }
        
        self._send_json_response(response, status_code)
    
    def _handle_health_status(self):
        """Detailed health status endpoint"""
        
        health = get_health_status()
        self._send_json_response(health.to_dict())
    
    def _handle_health_summary(self):
        """Health summary endpoint"""
        
        summary = get_health_summary()
        self._send_json_response(summary)
    
    def _handle_metrics(self):
        """Prometheus-style metrics endpoint"""
        
        health = get_health_status()
        monitor = get_health_monitor()
        
        metrics = []
        
        # System metrics
        metrics.append(f'rtai_cpu_percent {health.cpu_percent}')
        metrics.append(f'rtai_memory_percent {health.memory_percent}')
        metrics.append(f'rtai_disk_usage_percent {health.disk_usage_percent}')
        metrics.append(f'rtai_uptime_seconds {health.uptime_seconds}')
        
        # Health status as numeric
        status_map = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}
        metrics.append(f'rtai_health_status {status_map.get(health.status, 0)}')
        
        # Trade metrics
        metrics.append(f'rtai_indicators_calculated_total {health.indicators_calculated}')
        metrics.append(f'rtai_error_rate_5min {health.error_rate_5min}')
        metrics.append(f'rtai_feed_connected {int(health.feed_connected)}')
        
        # Connection count
        if health.active_connections >= 0:
            metrics.append(f'rtai_active_connections {health.active_connections}')
        
        response = '\n'.join(metrics) + '\n'
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        
        response = json.dumps(data, indent=2, default=str)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        
        response = {"error": message, "status_code": status_code}
        self._send_json_response(response, status_code)
    
    def log_message(self, format, *args):
        """Override to use structured logging"""
        
        structured_logger = get_structured_logger("health_dashboard")
        structured_logger.info(f"Health dashboard: {format % args}")


class HealthDashboard:
    """Health monitoring dashboard server"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
        self.structured_logger = get_structured_logger("health_dashboard")
    
    def start(self):
        """Start the health dashboard server"""
        
        if self.running:
            logger.warning("Health dashboard already running")
            return
        
        try:
            self.server = HTTPServer((self.host, self.port), HealthHandler)
            self.running = True
            
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            
            logger.success(f"🏥 Health dashboard started on http://{self.host}:{self.port}")
            self.structured_logger.info(
                f"Health dashboard started",
                host=self.host,
                port=self.port,
                endpoints=["/health", "/health/status", "/health/summary", "/metrics"]
            )
            
        except Exception as e:
            logger.error(f"Failed to start health dashboard: {e}")
            self.running = False
            raise
    
    def _run_server(self):
        """Run the HTTP server"""
        
        while self.running:
            try:
                self.server.serve_request()
            except Exception as e:
                if self.running:  # Don't log errors during shutdown
                    logger.error(f"Health dashboard server error: {e}")
                break
    
    def stop(self):
        """Stop the health dashboard server"""
        
        if not self.running:
            return
        
        self.running = False
        
        if self.server:
            self.server.server_close()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.info("🏥 Health dashboard stopped")
    
    def get_status_url(self) -> str:
        """Get the health status URL"""
        return f"http://{self.host}:{self.port}/health/status"
    
    def get_metrics_url(self) -> str:
        """Get the metrics URL"""
        return f"http://{self.host}:{self.port}/metrics"


# Global dashboard instance
_dashboard = None


def start_health_dashboard(host: str = "127.0.0.1", port: int = 8080) -> HealthDashboard:
    """Start the global health dashboard"""
    
    global _dashboard
    
    if _dashboard and _dashboard.running:
        logger.warning("Health dashboard already running")
        return _dashboard
    
    _dashboard = HealthDashboard(host, port)
    _dashboard.start()
    return _dashboard


def stop_health_dashboard():
    """Stop the global health dashboard"""
    
    global _dashboard
    
    if _dashboard:
        _dashboard.stop()
        _dashboard = None


def get_dashboard_urls() -> Dict[str, str]:
    """Get dashboard URLs if running"""
    
    if _dashboard and _dashboard.running:
        return {
            "health": f"http://{_dashboard.host}:{_dashboard.port}/health",
            "status": _dashboard.get_status_url(),
            "summary": f"http://{_dashboard.host}:{_dashboard.port}/health/summary",
            "metrics": _dashboard.get_metrics_url()
        }
    
    return {}


if __name__ == "__main__":
    # Test the health dashboard
    import time
    
    # Configure logging
    configure_json_logging("logs/health_dashboard_test.jsonl")
    
    # Start dashboard
    dashboard = start_health_dashboard(port=8081)
    
    try:
        print("🏥 Health dashboard running on http://127.0.0.1:8081")
        print("Endpoints:")
        print("  GET /health - Simple health check")  
        print("  GET /health/status - Detailed status")
        print("  GET /health/summary - Health summary")
        print("  GET /metrics - Prometheus metrics")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping health dashboard...")
        stop_health_dashboard()
        print("✅ Health dashboard stopped")
