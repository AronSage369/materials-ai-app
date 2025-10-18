import numpy as np
from typing import Dict, List, Any, Callable
import math

class MixturePredictor:
    def __init__(self):
        self.mixing_rules = {
            'linear': self.linear_mixing,
            'logarithmic': self.logarithmic_mixing,
            'geometric': self.geometric_mixing,
            'weighted_average': self.weighted_average_mixing
        }
        
        # Property-specific mixing rules
        self.property_mixing_methods = {
            'thermal_conductivity': 'weighted_average',
            'viscosity': 'logarithmic',
            'density': 'linear',
            'specific_heat': 'weighted_average',
            'surface_tension': 'geometric',
            'refractive_index': 'linear',
            'dielectric_constant': 'geometric',
            'solubility_parameter': 'linear',
            'boiling_point': 'weighted_average',
            'flash_point': 'weighted_average'
        }

    def predict_mixture_properties(self, formulation: Dict, 
                                 component_properties: Dict[str, float]) -> Dict[str, float]:
        """Predict mixture properties using appropriate mixing rules"""
        mixture_properties = {}
        composition = formulation.get('composition', [])
        
        if not composition:
            return component_properties
            
        # Get mass fractions and component properties
        mass_fractions = [comp.get('mass_percentage', 0) / 100 for comp in composition]
        
        for prop_name, component_value in component_properties.items():
            mixing_method = self.property_mixing_methods.get(prop_name, 'weighted_average')
            
            if mixing_method in self.mixing_rules:
                mixture_value = self.mixing_rules[mixing_method](
                    mass_fractions, [component_value] * len(composition)  # Simplified for demo
                )
                mixture_properties[prop_name] = mixture_value
            else:
                mixture_properties[prop_name] = component_value
                
        return mixture_properties

    def linear_mixing(self, fractions: List[float], values: List[float]) -> float:
        """Linear mixing rule: simple weighted average"""
        if len(fractions) != len(values):
            raise ValueError("Fractions and values must have same length")
            
        return sum(f * v for f, v in zip(fractions, values))

    def logarithmic_mixing(self, fractions: List[float], values: List[float]) -> float:
        """Logarithmic mixing rule for properties like viscosity"""
        if len(fractions) != len(values):
            raise ValueError("Fractions and values must have same length")
            
        # Avoid log(0) and handle negative values
        valid_values = [max(v, 1e-6) for v in values]
        log_sum = sum(f * math.log(v) for f, v in zip(fractions, valid_values))
        return math.exp(log_sum)

    def geometric_mixing(self, fractions: List[float], values: List[float]) -> float:
        """Geometric mean mixing rule"""
        if len(fractions) != len(values):
            raise ValueError("Fractions and values must have same length")
            
        product = 1.0
        for f, v in zip(fractions, values):
            product *= (v ** f)
        return product

    def weighted_average_mixing(self, fractions: List[float], values: List[float]) -> float:
        """Weighted average based on mass fractions"""
        return self.linear_mixing(fractions, values)

    def predict_mixture_boiling_point(self, formulation: Dict) -> float:
        """Specialized boiling point prediction for mixtures"""
        composition = formulation.get('composition', [])
        if len(composition) == 1:
            return composition[0].get('boiling_point', 100)
            
        # For mixtures, use weighted average of component boiling points
        bp_sum = 0
        total_mass = 0
        
        for comp in composition:
            mass_frac = comp.get('mass_percentage', 0) / 100
            bp = comp.get('boiling_point', 100)
            bp_sum += mass_frac * bp
            total_mass += mass_frac
            
        return bp_sum / total_mass if total_mass > 0 else 100

    def predict_azeotropic_behavior(self, formulation: Dict) -> Dict[str, Any]:
        """Predict potential azeotropic behavior in mixtures"""
        composition = formulation.get('composition', [])
        if len(composition) < 2:
            return {'is_azeotrope': False, 'confidence': 0.0}
            
        # Simple heuristic based on boiling point differences and polarity
        boiling_points = [comp.get('boiling_point', 100) for comp in composition]
        bp_range = max(boiling_points) - min(boiling_points)
        
        polarity_scores = []
        for comp in composition:
            smiles = comp.get('smiles', '').lower()
            polarity = 0.0
            if 'oh' in smiles or 'cooh' in smiles:
                polarity += 0.7
            if 'o' in smiles and 'c=o' in smiles:
                polarity += 0.5
            polarity_scores.append(polarity)
            
        polarity_similarity = 1.0 - (max(polarity_scores) - min(polarity_scores))
        
        # Azeotrope likelihood increases with similar boiling points and polarities
        azeotrope_prob = (1.0 - bp_range/100) * polarity_similarity
        is_azeotrope = azeotrope_prob > 0.6
        
        return {
            'is_azeotrope': is_azeotrope,
            'confidence': azeotrope_prob,
            'boiling_point_range': bp_range,
            'polarity_similarity': polarity_similarity
        }
