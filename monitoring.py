"""
Metrics and Monitoring Module for Void-State Tools

Provides Prometheus-compatible metrics export, health checks, and observability.

Usage:
    from void_state_tools.monitoring import MetricsCollector, start_metrics_server
    
    collector = MetricsCollector()
    start_metrics_server(port=9090)
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading


@dataclass
class Metric:
    """Base metric class"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""


@dataclass
class CounterMetric(Metric):
    """Counter metric (monotonically increasing)"""
    metric_type: str = "counter"


@dataclass
class GaugeMetric(Metric):
    """Gauge metric (can go up or down)"""
    metric_type: str = "gauge"


@dataclass
class HistogramMetric:
    """Histogram metric for distributions"""
    name: str
    buckets: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "histogram"


class MetricsCollector:
    """
    Collects and exports metrics in Prometheus format
    """
    
    def __init__(self):
        self.counters: Dict[str, CounterMetric] = {}
        self.gauges: Dict[str, GaugeMetric] = {}
        self.histograms: Dict[str, HistogramMetric] = {}
        self._lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize standard void-state metrics"""
        self.register_counter(
            "void_state_tools_total",
            "Total number of tools registered",
            {}
        )
        self.register_counter(
            "void_state_hook_executions_total",
            "Total number of hook executions",
            {"hook_name": ""}
        )
        self.register_gauge(
            "void_state_active_tools",
            "Number of currently active tools",
            {}
        )
        self.register_histogram(
            "void_state_hook_duration_seconds",
            "Hook execution duration in seconds",
            {"hook_name": ""},
            buckets=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        )
    
    def register_counter(self, name: str, help_text: str, labels: Dict[str, str]):
        """Register a new counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self.counters:
                self.counters[key] = CounterMetric(
                    name=name,
                    value=0.0,
                    labels=labels,
                    help_text=help_text
                )
    
    def register_gauge(self, name: str, help_text: str, labels: Dict[str, str]):
        """Register a new gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self.gauges:
                self.gauges[key] = GaugeMetric(
                    name=name,
                    value=0.0,
                    labels=labels,
                    help_text=help_text
                )
    
    def register_histogram(self, name: str, help_text: str, labels: Dict[str, str], 
                          buckets: List[float]):
        """Register a new histogram metric"""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self.histograms:
                histogram = HistogramMetric(name=name, labels=labels)
                for bucket in buckets:
                    histogram.buckets[bucket] = 0
                histogram.buckets[float('inf')] = 0
                self.histograms[key] = histogram
    
    def increment_counter(self, name: str, labels: Dict[str, str], value: float = 1.0):
        """Increment a counter"""
        with self._lock:
            key = self._make_key(name, labels)
            if key in self.counters:
                self.counters[key].value += value
    
    def set_gauge(self, name: str, labels: Dict[str, str], value: float):
        """Set a gauge value"""
        with self._lock:
            key = self._make_key(name, labels)
            if key in self.gauges:
                self.gauges[key].value = value
    
    def observe_histogram(self, name: str, labels: Dict[str, str], value: float):
        """Observe a value for histogram"""
        with self._lock:
            key = self._make_key(name, labels)
            if key in self.histograms:
                histogram = self.histograms[key]
                histogram.sum += value
                histogram.count += 1
                
                # Update buckets
                for bucket_limit in sorted(histogram.buckets.keys()):
                    if value <= bucket_limit:
                        histogram.buckets[bucket_limit] += 1
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key from name and labels"""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format"""
        output = []
        
        with self._lock:
            # Export counters
            for counter in self.counters.values():
                if counter.help_text:
                    output.append(f"# HELP {counter.name} {counter.help_text}")
                output.append(f"# TYPE {counter.name} counter")
                labels_str = self._format_labels(counter.labels)
                output.append(f"{counter.name}{labels_str} {counter.value}")
            
            # Export gauges
            for gauge in self.gauges.values():
                if gauge.help_text:
                    output.append(f"# HELP {gauge.name} {gauge.help_text}")
                output.append(f"# TYPE {gauge.name} gauge")
                labels_str = self._format_labels(gauge.labels)
                output.append(f"{gauge.name}{labels_str} {gauge.value}")
            
            # Export histograms
            for histogram in self.histograms.values():
                output.append(f"# TYPE {histogram.name} histogram")
                labels_str = self._format_labels(histogram.labels)
                
                for bucket_limit, count in sorted(histogram.buckets.items()):
                    bucket_labels = dict(histogram.labels, le=str(bucket_limit))
                    bucket_str = self._format_labels(bucket_labels)
                    output.append(f"{histogram.name}_bucket{bucket_str} {count}")
                
                output.append(f"{histogram.name}_sum{labels_str} {histogram.sum}")
                output.append(f"{histogram.name}_count{labels_str} {histogram.count}")
        
        return "\n".join(output)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels or all(v == "" for v in labels.values()):
            return ""
        
        valid_labels = {k: v for k, v in labels.items() if v}
        if not valid_labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in sorted(valid_labels.items())]
        return "{" + ",".join(label_pairs) + "}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            return {
                "counters": len(self.counters),
                "gauges": len(self.gauges),
                "histograms": len(self.histograms),
                "timestamp": time.time()
            }


class HealthCheck:
    """Health check for tools and system"""
    
    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.last_check_time: Dict[str, float] = {}
    
    def register_check(self, name: str, check_fn: callable):
        """Register a health check function"""
        self.checks[name] = check_fn
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        all_healthy = True
        
        for name, check_fn in self.checks.items():
            try:
                is_healthy = check_fn()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "timestamp": time.time()
                }
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
                all_healthy = False
            
            self.last_check_time[name] = time.time()
        
        return {
            "overall_status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": time.time()
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_server(port: int = 9090):
    """
    Start HTTP server for metrics export
    
    Args:
        port: Port to listen on
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        collector = get_metrics_collector()
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    metrics_text = collector.export_prometheus()
                    self.wfile.write(metrics_text.encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status": "healthy"}')
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        server = HTTPServer(("0.0.0.0", port), MetricsHandler)
        print(f"Metrics server started on port {port}")
        print(f"  Metrics: http://localhost:{port}/metrics")
        print(f"  Health:  http://localhost:{port}/health")
        
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        return server
    except ImportError:
        print("HTTP server not available in this environment")
        return None


__all__ = [
    "MetricsCollector",
    "HealthCheck",
    "get_metrics_collector",
    "start_metrics_server",
    "Metric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric"
]
