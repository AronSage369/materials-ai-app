import json
import numpy as np
import logging
import hashlib
import re
import os
import time
import gc
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple in-memory cache without diskcache dependency"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()

def cached(func):
    """Decorator for caching function results"""
    cache = SimpleCache()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Skip caching if explicitly disabled
        if kwargs.pop('no_cache', False):
            return func(*args, **kwargs)
            
        key = f"{func.__module__}.{func.__name__}_{cache.get_key((args, kwargs))}"
        
        result = cache.get(key)
        if result is not None:
            return result
        
        result = func(*args, **kwargs)
        cache.set(key, result)
        return result
    return wrapper

class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def cleanup_memory():
        """Perform memory cleanup"""
        gc.collect()
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get memory usage (simplified)"""
        return 0.0  # Simplified for now

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_formulation(formulation: Dict) -> Tuple[bool, str]:
        """Validate formulation structure"""
        required_keys = ['composition', 'agent_type']
        
        for key in required_keys:
            if key not in formulation:
                return False, f"Missing required key: {key}"
        
        if not isinstance(formulation['composition'], list):
            return False, "Composition must be a list"
        
        if len(formulation['composition']) == 0:
            return False, "Composition cannot be empty"
        
        total_percentage = sum(comp.get('mass_percentage', 0) for comp in formulation['composition'])
        if abs(total_percentage - 100.0) > 0.1:
            return False, f"Mass percentages must sum to 100%, got {total_percentage}"
        
        return True, "Valid"

    @staticmethod
    def validate_component(component: Dict) -> Tuple[bool, str]:
        """Validate component structure"""
        required_keys = ['name', 'mass_percentage']
        
        for key in required_keys:
            if key not in component:
                return False, f"Missing required key: {key}"
        
        mass_percentage = component.get('mass_percentage', 0)
        if not (0 <= mass_percentage <= 100):
            return False, f"Mass percentage must be between 0 and 100, got {mass_percentage}"
        
        return True, "Valid"

# Global instances
cache_manager = SimpleCache()
memory_manager = MemoryManager()
data_validator = DataValidator()
