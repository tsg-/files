#!/usr/bin/env python3
"""
KV Cache Monitor for vLLM
Monitors and displays KV cache usage statistics from Prometheus metrics endpoint

Usage:
    python3 monitor_kv_cache.py [--url URL] [--interval SECONDS] [--json]

Examples:
    # Monitor local vLLM instance
    python3 monitor_kv_cache.py

    # Monitor remote instance
    python3 monitor_kv_cache.py --url http://192.168.1.100:8000

    # Check every 10 seconds with JSON output
    python3 monitor_kv_cache.py --interval 10 --json
"""

import argparse
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


class KVCacheMonitor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.metrics_url = f"{self.base_url}/metrics"
        self.previous_stats = None

    def parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format"""
        metrics = {}

        for line in metrics_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                # Handle metrics with labels: metric_name{labels} value
                if '{' in line:
                    metric_part, value_part = line.rsplit(' ', 1)
                    # Extract metric name (before '{')
                    metric_name = metric_part.split('{')[0]
                    # Extract labels
                    labels_part = metric_part[metric_part.index('{')+1:metric_part.index('}')]
                    # Create key with labels for detailed tracking
                    full_key = f"{metric_name}{{{labels_part}}}"
                    metrics[full_key] = float(value_part)
                    # Also store without labels for easy access
                    if metric_name not in metrics:
                        metrics[metric_name] = float(value_part)
                else:
                    # Simple metric: metric_name value
                    metric_name, value_part = line.rsplit(' ', 1)
                    metrics[metric_name] = float(value_part)

            except Exception:
                continue

        return metrics

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch current stats from vLLM /metrics endpoint"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()

            # Parse Prometheus metrics
            raw_metrics = self.parse_prometheus_metrics(response.text)

            # Convert to friendly format
            # Note: vLLM metrics use percentage as 0-1 (0.5 = 50%), we convert to 0-100
            gpu_cache_pct = raw_metrics.get('vllm:gpu_cache_usage_perc', 0) * 100
            cpu_cache_pct = raw_metrics.get('vllm:cpu_cache_usage_perc', 0) * 100

            stats = {
                'gpu_cache_usage_perc': gpu_cache_pct,
                'cpu_cache_usage_perc': cpu_cache_pct,
                'num_running': int(raw_metrics.get('vllm:num_requests_running', 0)),
                'num_waiting': int(raw_metrics.get('vllm:num_requests_waiting', 0)),
                'num_swapped': int(raw_metrics.get('vllm:num_requests_swapped', 0)),
                'num_preemptions': int(raw_metrics.get('vllm:num_preemptions_total', 0)),
                'gpu_prefix_cache_hit_rate': raw_metrics.get('vllm:gpu_prefix_cache_hit_rate', 0) * 100,
                'cpu_prefix_cache_hit_rate': raw_metrics.get('vllm:cpu_prefix_cache_hit_rate', 0) * 100,
                'time_to_first_token_seconds': raw_metrics.get('vllm:time_to_first_token_seconds', 0),
                'time_per_output_token_seconds': raw_metrics.get('vllm:time_per_output_token_seconds', 0),
                'e2e_request_latency_seconds': raw_metrics.get('vllm:e2e_request_latency_seconds', 0),
                'raw_metrics': raw_metrics,
            }

            return stats

        except requests.exceptions.RequestException as e:
            print(f"Error fetching metrics: {e}", file=sys.stderr)
            return None

    def get_lmcache_metrics(self) -> Optional[Dict[str, Any]]:
        """Try to extract LMCache metrics from the metrics endpoint"""
        try:
            stats = self.get_stats()
            if not stats:
                return None

            raw_metrics = stats['raw_metrics']
            lmcache_metrics = {}

            # Look for lmcache-specific metrics
            for key, value in raw_metrics.items():
                if 'lmcache' in key.lower() or 'cache_hit' in key.lower():
                    lmcache_metrics[key] = value

            return lmcache_metrics if lmcache_metrics else None

        except Exception:
            return None

    def format_bytes(self, bytes_val: float) -> str:
        """Format bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    def format_duration(self, seconds: float) -> str:
        """Format duration to human-readable format"""
        if seconds < 0.001:
            return f"{seconds*1000000:.2f} μs"
        elif seconds < 1:
            return f"{seconds*1000:.2f} ms"
        else:
            return f"{seconds:.2f} s"

    def print_header(self):
        """Print monitoring header"""
        print("\n" + "=" * 80)
        print(" vLLM KV Cache Monitor (Prometheus Metrics)")
        print(f" Connected to: {self.base_url}")
        print(f" Metrics endpoint: {self.metrics_url}")
        print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def print_stats(self, stats: Dict[str, Any]):
        """Print formatted statistics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n[{timestamp}]")
        print("-" * 80)

        # GPU KV Cache Stats
        print("\n📊 GPU KV Cache Usage:")
        gpu_cache_pct = stats.get('gpu_cache_usage_perc', 0)
        print(f"  Usage:           {gpu_cache_pct:.2f}%")

        # Draw progress bar
        bar_width = 50
        filled = int(bar_width * gpu_cache_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}] {gpu_cache_pct:.1f}%")

        # GPU Prefix cache hit rate
        gpu_prefix_hit = stats.get('gpu_prefix_cache_hit_rate', 0)
        if gpu_prefix_hit > 0:
            print(f"  Prefix Hit Rate: {gpu_prefix_hit:.2f}%")

        # CPU KV Cache Stats
        cpu_cache_pct = stats.get('cpu_cache_usage_perc', 0)
        if cpu_cache_pct > 0:
            print(f"\n💾 CPU KV Cache Usage:")
            print(f"  Usage:           {cpu_cache_pct:.2f}%")

            cpu_prefix_hit = stats.get('cpu_prefix_cache_hit_rate', 0)
            if cpu_prefix_hit > 0:
                print(f"  Prefix Hit Rate: {cpu_prefix_hit:.2f}%")

        # Request Queue Stats
        print("\n🔄 Request Queue:")
        print(f"  Running:         {stats.get('num_running', 0)}")
        print(f"  Waiting:         {stats.get('num_waiting', 0)}")
        print(f"  Swapped:         {stats.get('num_swapped', 0)}")
        print(f"  Preemptions:     {stats.get('num_preemptions', 0)}")

        # Performance Metrics
        ttft = stats.get('time_to_first_token_seconds', 0)
        tpot = stats.get('time_per_output_token_seconds', 0)
        e2e = stats.get('e2e_request_latency_seconds', 0)

        if ttft > 0 or tpot > 0 or e2e > 0:
            print("\n⚡ Performance Metrics:")
            if ttft > 0:
                print(f"  TTFT (avg):      {self.format_duration(ttft)}")
            if tpot > 0:
                print(f"  TPOT (avg):      {self.format_duration(tpot)}")
            if e2e > 0:
                print(f"  E2E Latency:     {self.format_duration(e2e)}")

        # LMCache metrics
        lmcache_metrics = self.get_lmcache_metrics()
        if lmcache_metrics:
            print("\n💫 Cache Metrics:")
            for key, value in sorted(lmcache_metrics.items()):
                # Clean up metric name for display
                display_name = key.replace('vllm:', '').replace('_', ' ').title()
                if 'rate' in key.lower() or 'perc' in key.lower():
                    print(f"  {display_name:20s} {value*100:.2f}%")
                else:
                    print(f"  {display_name:20s} {value}")

        # Calculate changes from previous stats
        if self.previous_stats:
            print("\n📈 Changes Since Last Check:")

            prev_gpu = self.previous_stats.get('gpu_cache_usage_perc', 0)
            curr_gpu = gpu_cache_pct
            delta = curr_gpu - prev_gpu

            if abs(delta) > 0.01:
                symbol = "+" if delta > 0 else ""
                print(f"  GPU Cache:       {symbol}{delta:.2f}%")
            else:
                print(f"  GPU Cache:       No change")

            prev_running = self.previous_stats.get('num_running', 0)
            curr_running = stats.get('num_running', 0)
            if prev_running != curr_running:
                delta_running = curr_running - prev_running
                symbol = "+" if delta_running > 0 else ""
                print(f"  Running Reqs:    {symbol}{delta_running}")

        self.previous_stats = stats
        print("-" * 80)

    def print_json_stats(self, stats: Dict[str, Any]):
        """Print stats in JSON format"""
        # Remove raw_metrics for cleaner JSON output
        output_stats = {k: v for k, v in stats.items() if k != 'raw_metrics'}

        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': output_stats,
        }

        print(json.dumps(output, indent=2))

    def monitor(self, interval: int = 5, json_output: bool = False, continuous: bool = True):
        """Monitor KV cache usage"""
        if not json_output:
            self.print_header()

        try:
            while True:
                stats = self.get_stats()

                if stats is None:
                    if not continuous:
                        print("Failed to fetch stats. Server may be down or metrics not enabled.")
                        return False
                    time.sleep(interval)
                    continue

                if json_output:
                    self.print_json_stats(stats)
                else:
                    self.print_stats(stats)

                if not continuous:
                    return True

                time.sleep(interval)

        except KeyboardInterrupt:
            if not json_output:
                print("\n\n✓ Monitoring stopped by user")
            return True

    def check_health(self) -> bool:
        """Check if vLLM server is accessible"""
        try:
            # Try health endpoint first
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass

        # Fallback to metrics endpoint
        try:
            response = requests.get(self.metrics_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Monitor vLLM KV cache usage from Prometheus metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Monitor local vLLM instance
  %(prog)s --url http://10.0.0.1:8000   # Monitor remote instance
  %(prog)s --interval 10                # Check every 10 seconds
  %(prog)s --json                       # Output as JSON
  %(prog)s --once                       # Check once and exit

Note: vLLM must be started with --disable-log-stats=false to expose metrics
        """
    )

    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='vLLM server URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in seconds (default: 5)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output stats in JSON format'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Check once and exit (do not monitor continuously)'
    )

    args = parser.parse_args()

    # Create monitor
    monitor = KVCacheMonitor(args.url)

    # Check if server is accessible
    if not args.json:
        print(f"Checking connection to {args.url}...")

    if not monitor.check_health():
        print(f"Error: Cannot connect to vLLM server at {args.url}", file=sys.stderr)
        print("Make sure the server is running and accessible.", file=sys.stderr)
        print("\nTip: Ensure vLLM was started with metrics enabled.", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print("✓ Connected successfully\n")

    # Start monitoring
    success = monitor.monitor(
        interval=args.interval,
        json_output=args.json,
        continuous=not args.once
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
