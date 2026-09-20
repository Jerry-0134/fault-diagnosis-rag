import hashlib
import json
from datetime import datetime, timedelta

class MemoryCache:
    """内存缓存（纯Python实现，不需要Redis）"""
    
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, query):
        """生成缓存key"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query):
        """获取缓存"""
        key = self._get_key(query)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expire_at']:
                self.hits += 1
                print(f"缓存命中: {query[:30]}...")
                return entry['data']
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, query, data):
        """设置缓存"""
        key = self._get_key(query)
        self.cache[key] = {
            'data': data,
            'expire_at': datetime.now() + timedelta(seconds=self.ttl),
            'created_at': datetime.now()
        }
        print(f"缓存写入: {query[:30]}...")
    
    def stats(self):
        """缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'total_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}"
        }

# 全局缓存实例
cache = MemoryCache(ttl_seconds=3600)  # 1小时过期

