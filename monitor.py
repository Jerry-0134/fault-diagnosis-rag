import time
import functools
from datetime import datetime

response_times = []

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        response_times.append({
            'function': func.__name__,
            'time': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        print(f"{func.__name__} 耗时: {elapsed*1000:.2f}ms")
        return result
    return wrapper

def get_stats():
    if not response_times:
        return None
    times = [r['time'] for r in response_times]
    return {
        'avg_ms': sum(times) / len(times) * 1000,
        'p95_ms': sorted(times)[int(len(times)*0.95)] * 1000,
        'total': len(response_times)
    }
