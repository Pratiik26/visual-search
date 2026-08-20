"""
In-Memory Fast LRU Cache Utilities for Query Vectors & Image Payloads
"""

from collections import OrderedDict
from typing import Any, Optional


class LRUFeatureCache:
    """Fast LRU Memory Cache for Query Embeddings & Saliency Analysis"""

    def __init__(self, capacity: int = 256):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)
