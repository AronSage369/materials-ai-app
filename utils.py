import json
import numpy as np
import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
import diskcache
from functools import wraps
import psutil
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('materials_discovery.log'),
        logging.StreamHandler()
    ]
)

# Reduce noise from external libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)

class CacheManager:
    """Enhanced cache manager with disk persistence"""
    def __init__(self, cache_dir="./cache", max_size=1000):
        self.cache = diskcache.Cache(cache_dir, size_limit=max_size * 1024 * 1024)  # MB
        self.logger = logging.getLogger(__name__)
    
    def get_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                self.logger.debug(f"Cache hit for key: {key[:8]}...")
                return self.cache[key]
            return None
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration"""
        try:
            self.cache.set(key, value, expire=expire)
            self.logger.debug(f"Cache set for key: {key[:8]}...")
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
    
    def clear(self):
        """Clear cache"""
        try:
            self.cache.clear()
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")

def cached(func):
    """Decorator for caching function results"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Skip caching if explicitly disabled
        if kwargs.pop('no_cache', False):
            return func(*args, **kwargs)
            
        cache = CacheManager()
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
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def cleanup_memory():
        """Perform memory cleanup"""
        initial_memory = MemoryManager.get_memory_usage()
        gc.collect()
        final_memory = MemoryManager.get_memory_usage()
        logging.info(f"Memory cleanup: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
        return final_memory
    
    @staticmethod
    def check_memory_limit(limit_mb: float = 1000) -> bool:
        """Check if memory usage exceeds limit"""
        return MemoryManager.get_memory_usage() > limit_mb

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
        
        for i, comp in enumerate(formulation['composition']):
            is_valid, message = DataValidator.validate_component(comp)
            if not is_valid:
                return False, f"Component {i}: {message}"
                
        return True, "Valid"
    
    @staticmethod
    def validate_component(component: Dict) -> Tuple[bool, str]:
        """Validate component structure"""
        required = ['name', 'mass_percentage']
        
        for key in required:
            if key not in component:
                return False, f"Missing required key: {key}"
        
        mass_pct = component['mass_percentage']
        if not (0 <= mass_pct <= 100):
            return False, f"Mass percentage must be between 0 and 100, got {mass_pct}"
            
        return True, "Valid"
    
    @staticmethod
    def normalize_percentages(composition: List[Dict]) -> List[Dict]:
        """Normalize mass percentages to sum to 100"""
        total = sum(comp.get('mass_percentage', 0) for comp in composition)
        
        if total == 0:
            # Equal distribution if all zero
            equal_pct = 100.0 / len(composition)
            for comp in composition:
                comp['mass_percentage'] = equal_pct
        else:
            # Normalize to 100
            for comp in composition:
                comp['mass_percentage'] = (comp['mass_percentage'] / total) * 100
                
        return composition

class PropertyCalculator:
    """Advanced property calculation utilities"""
    
    @staticmethod
    def calculate_mixture_property(components: List[Dict], property_name: str, 
                                 method: str = 'weighted_average') -> float:
        """Calculate mixture property using specified method"""
        if not components:
            return 0.0
        
        values = []
        weights = []
        
        for comp in components:
            mass_frac = comp.get('mass_percentage', 0) / 100
            prop_value = comp.get(property_name, 0)
            
            if prop_value is not None and mass_frac > 0:
                values.append(prop_value)
                weights.append(mass_frac)
        
        if not values:
            return 0.0
        
        if method == 'weighted_average':
            return np.average(values, weights=weights)
        elif method == 'geometric_mean':
            log_sum = sum(w * np.log(max(v, 1e-6)) for w, v in zip(weights, values))
            return np.exp(log_sum)
        elif method == 'harmonic_mean':
            return sum(weights) / sum(w / max(v, 1e-6) for w, v in zip(weights, values))
        else:
            return np.average(values, weights=weights)
    
    @staticmethod
    def estimate_property_from_smiles(smiles: str, property_type: str) -> float:
        """Estimate property from SMILES string using simple rules"""
        if not smiles:
            return 0.0
        
        smiles_lower = smiles.lower()
        
        if property_type == 'molecular_weight':
            # Very rough estimate from atom counts
            atoms = re.findall(r'[A-Z][a-z]?', smiles.upper())
            atom_weights = {'C': 12, 'H': 1, 'O': 16, 'N': 14, 'S': 32, 'P': 31, 
                           'F': 19, 'Cl': 35.5, 'Br': 80, 'I': 127}
            
            weight = 0
            for atom in atoms:
                weight += atom_weights.get(atom, 12)  # Default to carbon
            
            return max(weight, 18)  # Minimum water weight
        
        elif property_type == 'polarity':
            # Estimate polarity from functional groups
            polar_groups = ['oh', 'cooh', 'c=o', 'n', 'o', 'f', 'cl']
            polarity_score = 0
            
            for group in polar_groups:
                if group in smiles_lower:
                    polarity_score += 0.2
            
            return min(polarity_score, 1.0)
        
        else:
            return 0.0

class ResultsAnalyzer:
    """Advanced results analysis utilities"""
    
    @staticmethod
    def calculate_diversity_metric(formulations: List[Dict]) -> float:
        """Calculate diversity metric for formulation set"""
        if len(formulations) <= 1:
            return 1.0
        
        # Compare compositions using Jaccard similarity
        similarities = []
        
        for i in range(len(formulations)):
            for j in range(i + 1, len(formulations)):
                comp1 = set(comp.get('cid', comp.get('name')) for comp in formulations[i]['composition'])
                comp2 = set(comp.get('cid', comp.get('name')) for comp in formulations[j]['composition'])
                
                if comp1 and comp2:
                    similarity = len(comp1.intersection(comp2)) / len(comp1.union(comp2))
                    similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity  # Diversity = 1 - similarity
        else:
            return 1.0
    
    @staticmethod
    def find_pareto_front(formulations: List[Dict], objectives: List[str]) -> List[Dict]:
        """Find Pareto-optimal formulations for multiple objectives"""
        if not formulations or not objectives:
            return formulations
        
        pareto_front = []
        
        for formulation in formulations:
            dominated = False
            properties = formulation.get('predicted_properties', {})
            
            for other in formulations:
                if formulation == other:
                    continue
                    
                other_props = other.get('predicted_properties', {})
                dominates = True
                
                for obj in objectives:
                    if obj in properties and obj in other_props:
                        if other_props[obj] <= properties[obj]:
                            dominates = False
                            break
                
                if dominates:
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(formulation)
        
        return pareto_front

# Global instances
cache_manager = CacheManager()
memory_manager = MemoryManager()
data_validator = DataValidator()
property_calculator = PropertyCalculator()
results_analyzer = ResultsAnalyzer()
