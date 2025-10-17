# mixture_predictor.py - Advanced mixture property prediction
import numpy as np
from typing import Dict, List, Any
import streamlit as st

class MixturePredictor:
    def __init__(self):
        self.mixing_rules = {
            'linear': ['density', 'specific_heat', 'refractive_index', 'dielectric_constant'],
            'exponential': ['viscosity', 'diffusion_coefficient'],
            'geometric': ['thermal_conductivity', 'surface_tension'],
            'complex': ['adsorption_capacity', 'catalytic_activity', 'solubility']
        }
        
        # Property estimation parameters
        self.property_estimators = {
            'thermal_conductivity': self.estimate_thermal_conductivity,
            'viscosity': self.estimate_viscosity,
            'flash_point': self.estimate_flash_point,
            'specific_heat': self.estimate_specific_heat,
            'surface_area': self.estimate_surface_area
        }
    
    def predict_all_properties(self, formulations: List[Dict], strategy: Dict) -> List[Dict]:
        """Predict properties for all formulations"""
        target_properties = strategy.get('target_properties', {})
        
        for formulation in formulations:
            compounds = formulation.get('compounds', [])
            ratios = formulation.get('ratios', [1.0] * len(compounds))
            
            if not compounds:
                continue
                
            predicted_properties = {}
            
            for prop_name in target_properties.keys():
                # Predict property for mixture
                prediction = self.predict_mixture_property(compounds, ratios, prop_name)
                predicted_properties[prop_name] = prediction
            
            formulation['predicted_properties'] = predicted_properties
        
        return formulations
    
    def predict_mixture_property(self, compounds: List, ratios: List[float], property_name: str) -> Dict[str, Any]:
        """Predict a specific property for a mixture"""
        
        # Select appropriate mixing rule
        mixing_rule = self.select_mixing_rule(property_name, compounds)
        
        # Get individual compound property estimates
        individual_props = []
        confidences = []
        
        for compound in compounds:
            prop_value, confidence = self.estimate_compound_property(compound, property_name)
            individual_props.append(prop_value)
            confidences.append(confidence)
        
        # Apply mixing rule
        if mixing_rule == 'linear':
            mixture_value = self.linear_mixing(individual_props, ratios)
        elif mixing_rule == 'geometric':
            mixture_value = self.geometric_mixing(individual_props, ratios)
        elif mixing_rule == 'exponential':
            mixture_value = self.exponential_mixing(individual_props, ratios)
        else:  # complex or default to linear
            mixture_value = self.linear_mixing(individual_props, ratios)
        
        # Calculate overall confidence
        avg_confidence = np.mean(confidences) if confidences else 0.5
        rule_confidence = self.get_mixing_rule_confidence(mixing_rule, property_name)
        overall_confidence = avg_confidence * rule_confidence
        
        return {
            'value': mixture_value,
            'confidence': overall_confidence,
            'mixing_rule_used': mixing_rule,
            'individual_values': individual_props,
            'unit': self.get_property_unit(property_name)
        }
    
    def select_mixing_rule(self, property_name: str, compounds: List) -> str:
        """Select appropriate mixing rule for property"""
        for rule_type, properties in self.mixing_rules.items():
            if property_name in properties:
                return rule_type
        
        # Default rules based on property name patterns
        if any(term in property_name for term in ['viscosity', 'diffusion']):
            return 'exponential'
        elif any(term in property_name for term in ['conductivity', 'tension']):
            return 'geometric'
        else:
            return 'linear'
    
    def estimate_compound_property(self, compound, property_name: str) -> tuple:
        """Estimate property for individual compound"""
        estimator = self.property_estimators.get(property_name)
        
        if estimator:
            return estimator(compound)
        else:
            # Default estimation
            return self.default_property_estimation(compound, property_name)
    
    def estimate_thermal_conductivity(self, compound) -> tuple:
        """Estimate thermal conductivity in W/m·K"""
        base_value = 0.08  # Base for organic liquids
        
        # Molecular weight effect - medium MW optimal
        if compound.molecular_weight:
            mw_factor = 1 - abs(compound.molecular_weight - 300) / 1000
            base_value += mw_factor * 0.05
        
        # Structural effects
        name = str(compound.iupac_name).lower() if compound.iupac_name else ""
        
        # Silicones have better thermal conductivity
        if any(term in name for term in ['siloxane', 'silicone']):
            base_value += 0.04
        
        # Aromatics vs aliphatics
        if 'benzene' in name or 'phenyl' in name:
            base_value += 0.02  # Aromatics slightly better
        
        confidence = 0.6  # Moderate confidence for estimations
        
        return max(0.05, min(0.2, base_value)), confidence
    
    def estimate_viscosity(self, compound) -> tuple:
        """Estimate viscosity in cSt at 100°C"""
        if not compound.molecular_weight:
            return 10.0, 0.3
        
        # Base viscosity estimation from molecular weight
        base_viscosity = compound.molecular_weight / 50
        
        # Structural corrections
        name = str(compound.iupac_name).lower() if compound.iupac_name else ""
        
        # Branched molecules have lower viscosity
        if any(term in name for term in ['branched', 'iso', 'neo']):
            base_viscosity *= 0.7
        
        # Cyclic structures increase viscosity
        if any(term in name for term in ['cyclo', 'cyclic']):
            base_viscosity *= 1.3
        
        # Aromatics increase viscosity
        if any(term in name for term in ['benzene', 'phenyl', 'aromatic']):
            base_viscosity *= 1.4
        
        confidence = 0.7
        
        return max(1.0, min(100.0, base_viscosity)), confidence
    
    def estimate_flash_point(self, compound) -> tuple:
        """Estimate flash point in °C"""
        if not compound.molecular_weight:
            return 150.0, 0.4
        
        # Basic correlation with molecular weight
        flash_point = 50 + (compound.molecular_weight * 0.25)
        
        # Structural effects
        name = str(compound.iupac_name).lower() if compound.iupac_name else ""
        
        # Oxygenated compounds often have higher flash points
        if any(term in name for term in ['alcohol', 'glycol', 'ester', 'ether']):
            flash_point += 20
        
        # Highly branched may have lower flash points
        if any(term in name for term in ['branched', 'iso']):
            flash_point -= 10
        
        confidence = 0.65
        
        return max(50.0, min(300.0, flash_point)), confidence
    
    def estimate_specific_heat(self, compound) -> tuple:
        """Estimate specific heat in J/kg·K"""
        base_value = 1800  # Base for organic compounds
        
        # Molecular weight effect - lighter molecules often have higher specific heat
        if compound.molecular_weight:
            mw_factor = max(0.5, 1000 / compound.molecular_weight)
            base_value *= mw_factor
        
        # Oxygenated compounds often have higher specific heat
        name = str(compound.iupac_name).lower() if compound.iupac_name else ""
        if any(term in name for term in ['alcohol', 'glycol', 'ester']):
            base_value += 300
        
        confidence = 0.6
        
        return max(1000, min(3000, base_value)), confidence
    
    def estimate_surface_area(self, compound) -> tuple:
        """Estimate surface area in m²/g for porous materials"""
        name = str(compound.iupac_name).lower() if compound.iupac_name else ""
        
        # Check if compound is likely porous
        if any(term in name for term in ['zeolite', 'MOF', 'porous', 'activated']):
            # Porous materials have high surface area
            surface_area = 500 + (hash(name) % 1500)  # Pseudo-random based on name
            confidence = 0.7
        else:
            # Non-porous materials have low surface area
            surface_area = 5 + (hash(name) % 20)
            confidence = 0.8
        
        return max(1, min(2000, surface_area)), confidence
    
    def default_property_estimation(self, compound, property_name: str) -> tuple:
        """Default property estimation when no specific estimator exists"""
        # Simple estimation based on molecular weight
        if not compound.molecular_weight:
            return 1.0, 0.3
        
        normalized_value = min(1.0, compound.molecular_weight / 1000)
        
        # Map to reasonable ranges based on property type
        property_ranges = {
            'efficiency': (0.1, 0.9),
            'stability': (0.5, 0.95),
            'selectivity': (0.3, 0.98),
            'capacity': (1, 10)
        }
        
        for prop_pattern, (min_val, max_val) in property_ranges.items():
            if prop_pattern in property_name:
                value = min_val + normalized_value * (max_val - min_val)
                return value, 0.4
        
        # Default range
        return 0.5 + normalized_value * 0.4, 0.3
    
    # Mixing rules implementation
    def linear_mixing(self, individual_values: List[float], ratios: List[float]) -> float:
        """Linear mixing rule: weighted average"""
        return sum(val * ratio for val, ratio in zip(individual_values, ratios))
    
    def geometric_mixing(self, individual_values: List[float], ratios: List[float]) -> float:
        """Geometric mixing rule: product of values raised to ratios"""
        if any(val <= 0 for val in individual_values):
            return self.linear_mixing(individual_values, ratios)
        
        log_sum = sum(np.log(val) * ratio for val, ratio in zip(individual_values, ratios))
        return np.exp(log_sum)
    
    def exponential_mixing(self, individual_values: List[float], ratios: List[float]) -> float:
        """Exponential mixing rule for properties like viscosity"""
        if any(val <= 0 for val in individual_values):
            return self.linear_mixing(individual_values, ratios)
        
        # For viscosity-like properties, often use log mixing
        log_sum = sum(np.log(val) * ratio for val, ratio in zip(individual_values, ratios))
        return np.exp(log_sum)
    
    def get_mixing_rule_confidence(self, mixing_rule: str, property_name: str) -> float:
        """Get confidence level for mixing rule"""
        rule_confidences = {
            'linear': 0.8,
            'geometric': 0.7,
            'exponential': 0.6,
            'complex': 0.4
        }
        return rule_confidences.get(mixing_rule, 0.5)
    
    def get_property_unit(self, property_name: str) -> str:
        """Get unit for property"""
        units = {
            'thermal_conductivity': 'W/m·K',
            'viscosity': 'cSt',
            'flash_point': '°C',
            'specific_heat': 'J/kg·K',
            'surface_area': 'm²/g',
            'density': 'g/cm³',
            'refractive_index': '',
            'dielectric_constant': ''
        }
        return units.get(property_name, '')
