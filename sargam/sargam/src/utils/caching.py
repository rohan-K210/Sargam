import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Optional
import threading


class ModelCache:
    """Thread-safe model cache with disk persistence."""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._lock = threading.Lock()
    
    def _get_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get from memory or disk cache."""
        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]
            
            cache_file = self.cache_dir / f"{self._get_key(key)}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    self._memory_cache[key] = value
                    return value
                except Exception:
                    pass
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set in both memory and disk cache."""
        with self._lock:
            self._memory_cache[key] = value
            cache_file = self.cache_dir / f"{self._get_key(key)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception:
                pass
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()


def cached(cache: ModelCache, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "|".join(key_parts)
            
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator


# Global cache instance
model_cache = ModelCache()