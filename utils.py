import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class FormulationResult:
    composition: List[Dict]
    properties: Dict[str, float]
    score: float
    risk: float
    agent_type: str
    reasoning: str

class CacheManager:
    """Simple cache manager for API responses and calculations"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True)
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

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_formulation(formulation: Dict) -> bool:
        """Validate formulation structure"""
        required_keys = ['composition', 'agent_type']
        
        if not all(key in formulation for key in required_keys):
            return False
        
        if not isinstance(formulation['composition'], list):
            return False
        
        for comp in formulation['composition']:
            if not DataValidator.validate_component(comp):
                return False
                
        return True
    
    @staticmethod
    def validate_component(component: Dict) -> bool:
        """Validate component structure"""
        required = ['name', 'mass_percentage']
        
        if not all(key in component for key in required):
            return False
        
        if not (0 <= component['mass_percentage'] <= 100):
            return False
            
        return True
    
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
        
        smiles = smiles.lower()
        
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
                if group in smiles:
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
                comp1 = set(comp['cid'] for comp in formulations[i]['composition'])
                comp2 = set(comp['cid'] for comp in formulations[j]['composition'])
                
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
    
    @staticmethod
    def calculate_robustness_score(formulation: Dict, uncertainty: float = 0.1) -> float:
        """Calculate robustness score considering property uncertainties"""
        properties = formulation.get('predicted_properties', {})
        strategy = formulation.get('strategy_alignment', {})
        
        if not properties or not strategy:
            return 0.5
        
        # Simulate property variations and check constraints
        num_samples = 100
        feasible_count = 0
        
        for _ in range(num_samples):
            feasible = True
            
            for prop_name, target_info in strategy.get('target_properties', {}).items():
                if prop_name in properties:
                    base_value = properties[prop_name]
                    # Add random noise
                    varied_value = base_value * (1 + np.random.normal(0, uncertainty))
                    
                    # Check constraints
                    min_val = target_info.get('min')
                    max_val = target_info.get('max')
                    
                    if min_val and varied_value < min_val:
                        feasible = False
                        break
                    if max_val and varied_value > max_val:
                        feasible = False
                        break
            
            if feasible:
                feasible_count += 1
        
        return feasible_count / num_samples

# Global instances
cache_manager = CacheManager()
data_validator = DataValidator()
property_calculator = PropertyCalculator()
results_analyzer = ResultsAnalyzer()
