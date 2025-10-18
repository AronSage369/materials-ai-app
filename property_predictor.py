import numpy as np
from typing import Dict, List, Any, Optional
import re

class AdvancedPropertyPredictor:
    def __init__(self):
        self.property_models = {
            'thermal_conductivity': self.predict_thermal_conductivity,
            'viscosity': self.predict_viscosity,
            'boiling_point': self.predict_boiling_point,
            'flash_point': self.predict_flash_point,
            'specific_heat': self.predict_specific_heat,
            'density': self.predict_density,
            'surface_tension': self.predict_surface_tension,
            'refractive_index': self.predict_refractive_index,
            'dielectric_constant': self.predict_dielectric_constant,
            'solubility_parameter': self.predict_solubility_parameter
        }
        
    def predict_all_properties(self, formulations: List[Dict], 
                         target_properties: Dict) -> List[Dict]:
    """Predict all required properties for formulations"""
    for formulation in formulations:
        predicted_properties = {}
        confidence_scores = {}
        
        for prop_name in target_properties.keys():
            if prop_name in self.property_models:
                try:
                    value, confidence = self.property_models[prop_name](formulation)
                    # Ensure we return a numeric value
                    if value is None or not isinstance(value, (int, float)):
                        value = 0.0
                        confidence = 0.1
                    predicted_properties[prop_name] = value
                    confidence_scores[prop_name] = confidence
                except Exception as e:
                    print(f"Error predicting {prop_name}: {e}")
                    predicted_properties[prop_name] = 0.0
                    confidence_scores[prop_name] = 0.0
        
        formulation['predicted_properties'] = predicted_properties
        formulation['prediction_confidence'] = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
        
    return formulations

    def predict_thermal_conductivity(self, formulation: Dict) -> tuple[float, float]:
        """Predict thermal conductivity in W/m·K"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Lower MW generally higher thermal conductivity
        base_value = 0.1 + (500 - avg_mw) / 1000
        base_value = max(0.05, min(0.8, base_value))
        
        # Adjust for composition complexity
        complexity_factor = self._get_complexity_factor(formulation)
        final_value = base_value * (1.0 + complexity_factor * 0.2)
        
        confidence = 0.7
        return final_value, confidence

    def predict_viscosity(self, formulation: Dict) -> tuple[float, float]:
        """Predict viscosity in cP"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Higher MW generally higher viscosity
        base_value = 0.1 + (avg_mw - 50) / 200
        base_value = max(0.1, min(1000, base_value))
        
        # Adjust for molecular complexity
        complexity_factor = self._get_complexity_factor(formulation)
        final_value = base_value * (1.0 + complexity_factor * 0.5)
        
        confidence = 0.8
        return final_value, confidence

    def predict_boiling_point(self, formulation: Dict) -> tuple[float, float]:
        """Predict boiling point in °C"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Boiling point generally increases with MW
        base_value = 50 + (avg_mw - 50) * 2.0
        base_value = max(-100, min(400, base_value))
        
        # Adjust for polarity and hydrogen bonding
        polarity_factor = self._get_polarity_factor(formulation)
        final_value = base_value * (1.0 + polarity_factor * 0.3)
        
        confidence = 0.75
        return final_value, confidence

    def predict_flash_point(self, formulation: Dict) -> tuple[float, float]:
        """Predict flash point in °C"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Flash point generally increases with MW
        base_value = 30 + (avg_mw - 50) * 1.5
        base_value = max(-50, min(300, base_value))
        
        confidence = 0.65
        return base_value, confidence

    def predict_specific_heat(self, formulation: Dict) -> tuple[float, float]:
        """Predict specific heat in J/g·K"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Lower MW generally higher specific heat
        base_value = 1.0 + (200 - avg_mw) / 500
        base_value = max(0.5, min(4.0, base_value))
        
        confidence = 0.7
        return base_value, confidence

    def predict_density(self, formulation: Dict) -> tuple[float, float]:
        """Predict density in g/mL"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Density increases with MW but plateaus
        base_value = 0.7 + (avg_mw - 50) / 1000
        base_value = max(0.6, min(1.5, base_value))
        
        confidence = 0.8
        return base_value, confidence

    def predict_surface_tension(self, formulation: Dict) -> tuple[float, float]:
        """Predict surface tension in mN/m"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Surface tension increases with MW
        base_value = 15 + (avg_mw - 50) / 20
        base_value = max(10, min(80, base_value))
        
        confidence = 0.6
        return base_value, confidence

    def predict_refractive_index(self, formulation: Dict) -> tuple[float, float]:
        """Predict refractive index"""
        avg_mw = self._get_average_molecular_weight(formulation)
        
        # Heuristic: Refractive index slightly increases with MW
        base_value = 1.3 + (avg_mw - 50) / 2000
        base_value = max(1.3, min(1.7, base_value))
        
        confidence = 0.7
        return base_value, confidence

    def predict_dielectric_constant(self, formulation: Dict) -> tuple[float, float]:
        """Predict dielectric constant"""
        polarity_factor = self._get_polarity_factor(formulation)
        
        # Heuristic: Higher polarity = higher dielectric constant
        base_value = 2.0 + polarity_factor * 30.0
        base_value = max(2.0, min(80.0, base_value))
        
        confidence = 0.65
        return base_value, confidence

    def predict_solubility_parameter(self, formulation: Dict) -> tuple[float, float]:
        """Predict Hansen solubility parameter in MPa^1/2"""
        avg_mw = self._get_average_molecular_weight(formulation)
        polarity_factor = self._get_polarity_factor(formulation)
        
        # Heuristic: Combination of MW and polarity effects
        base_value = 15 + (avg_mw - 100) / 100 + polarity_factor * 5
        base_value = max(10, min(30, base_value))
        
        confidence = 0.6
        return base_value, confidence

    def _get_average_molecular_weight(self, formulation: Dict) -> float:
        """Calculate average molecular weight of formulation"""
        total_mass = 0
        weighted_mw = 0
        
        for component in formulation.get('composition', []):
            mass_frac = component.get('mass_percentage', 0) / 100
            mw = self._get_mw(component)
            weighted_mw += mass_frac * mw
            total_mass += mass_frac
            
        return weighted_mw / total_mass if total_mass > 0 else 100

    def _get_mw(self, compound: Dict) -> float:
        """Safely get molecular weight with robust error handling"""
        try:
            mw = compound.get('molecular_weight')
            if mw is None:
                return 100.0  # Default fallback
            
            # Handle string values
            if isinstance(mw, str):
                mw = float(mw.strip()) if mw.strip() else 100.0
            
            # Validate range
            if not (0.1 <= mw <= 100000):
                return 100.0
                
            return float(mw)
        except (TypeError, ValueError, AttributeError):
            return 100.0  # Comprehensive fallback

    def _get_complexity_factor(self, formulation: Dict) -> float:
        """Calculate molecular complexity factor (0-1 scale)"""
        complexities = []
        
        for component in formulation.get('composition', []):
            # Estimate complexity from SMILES string length and patterns
            smiles = component.get('smiles', '')
            if smiles:
                # Simple heuristic: longer SMILES = more complex
                comp_score = min(len(smiles) / 50, 1.0)
                complexities.append(comp_score)
            else:
                complexities.append(0.5)  # Default medium complexity
        
        return np.mean(complexities) if complexities else 0.5

    def _get_polarity_factor(self, formulation: Dict) -> float:
        """Calculate polarity factor based on functional groups"""
        polarity_scores = []
        
        for component in formulation.get('composition', []):
            smiles = component.get('smiles', '').lower()
            score = 0.0
            
            # Detect polar functional groups
            polar_groups = [
                ('o', 0.3),  # Oxygen atoms
                ('n', 0.4),  # Nitrogen atoms  
                ('oh', 0.7), # Hydroxyl groups
                ('c=o', 0.6), # Carbonyl groups
                ('coo', 0.5), # Ester/carboxyl
                ('cn', 0.4),  # Nitrile
                ('cl', 0.2),  # Chlorine
                ('f', 0.3),   # Fluorine
            ]
            
            for group, group_score in polar_groups:
                if group in smiles:
                    score = max(score, group_score)
            
            polarity_scores.append(score)
        
        return np.mean(polarity_scores) if polarity_scores else 0.3
