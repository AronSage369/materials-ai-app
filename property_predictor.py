import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
import random
import logging

class AdvancedPropertyPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
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
            'solubility_parameter': self.predict_solubility_parameter,
            'absorption_capacity': self.predict_absorption_capacity,
            'thermal_stability': self.predict_thermal_stability,
            'toxicity': self.predict_toxicity
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
                        predicted_properties[prop_name] = value
                        confidence_scores[prop_name] = confidence
                    except Exception as e:
                        self.logger.error(f"Error predicting {prop_name}: {e}")
                        predicted_properties[prop_name] = 0.0
                        confidence_scores[prop_name] = 0.0
            
            formulation['predicted_properties'] = predicted_properties
            formulation['prediction_confidence'] = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
            
        return formulations

    def predict_thermal_conductivity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict thermal conductivity in W/m·K using enhanced heuristics"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            
            # Enhanced heuristic considering multiple factors
            base_value = 0.15 + (500 - avg_mw) / 1500
            # Polar molecules often have higher thermal conductivity
            polarity_boost = polarity * 0.3
            final_value = max(0.05, min(0.8, base_value + polarity_boost))
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in thermal conductivity prediction: {e}")
            return 0.2, 0.3

    def predict_viscosity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict viscosity in cP using molecular analysis"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            complexity = self._get_complexity_factor(formulation)
            
            # Enhanced viscosity model
            base_value = 1.0 + (avg_mw - 50) / 150
            complexity_effect = complexity * 0.8
            final_value = max(0.5, min(500, base_value + complexity_effect))
            
            confidence = 0.75
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in viscosity prediction: {e}")
            return 10.0, 0.3

    def predict_boiling_point(self, formulation: Dict) -> Tuple[float, float]:
        """Predict boiling point in °C using advanced heuristics"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            h_bonding = self._get_hydrogen_bonding_factor(formulation)
            
            # Multi-factor boiling point estimation
            base_value = 50 + (avg_mw - 50) * 1.8
            polarity_effect = polarity * 40
            hbond_effect = h_bonding * 60
            final_value = max(-50, min(400, base_value + polarity_effect + hbond_effect))
            
            confidence = 0.8
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in boiling point prediction: {e}")
            return 100.0, 0.3

    def predict_flash_point(self, formulation: Dict) -> Tuple[float, float]:
        """Predict flash point in °C"""
        try:
            boiling_point, _ = self.predict_boiling_point(formulation)
            # Flash point is typically 10-40°C below boiling point
            flash_point = boiling_point - 25
            flash_point = max(-50, min(300, flash_point))
            
            confidence = 0.65
            return flash_point, confidence
        except Exception as e:
            self.logger.error(f"Error in flash point prediction: {e}")
            return 50.0, 0.3

    def predict_specific_heat(self, formulation: Dict) -> Tuple[float, float]:
        """Predict specific heat in J/g·K"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            
            # Polar molecules often have higher specific heat
            base_value = 1.5 + (200 - avg_mw) / 400
            polarity_boost = polarity * 0.5
            final_value = max(0.8, min(4.0, base_value + polarity_boost))
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in specific heat prediction: {e}")
            return 2.0, 0.3

    def predict_density(self, formulation: Dict) -> Tuple[float, float]:
        """Predict density in g/mL"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            complexity = self._get_complexity_factor(formulation)
            
            base_value = 0.8 + (avg_mw - 50) / 800
            # Complex molecules might be more dense
            complexity_effect = complexity * 0.3
            final_value = max(0.7, min(1.8, base_value + complexity_effect))
            
            confidence = 0.8
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in density prediction: {e}")
            return 1.0, 0.3

    def predict_surface_tension(self, formulation: Dict) -> Tuple[float, float]:
        """Predict surface tension in mN/m"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            
            base_value = 20 + (avg_mw - 50) / 15
            polarity_effect = polarity * 25
            final_value = max(15, min(80, base_value + polarity_effect))
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in surface tension prediction: {e}")
            return 30.0, 0.3

    def predict_refractive_index(self, formulation: Dict) -> Tuple[float, float]:
        """Predict refractive index"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            
            base_value = 1.35 + (avg_mw - 50) / 2500
            polarity_effect = polarity * 0.2
            final_value = max(1.3, min(1.7, base_value + polarity_effect))
            
            confidence = 0.75
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in refractive index prediction: {e}")
            return 1.4, 0.3

    def predict_dielectric_constant(self, formulation: Dict) -> Tuple[float, float]:
        """Predict dielectric constant"""
        try:
            polarity = self._get_polarity_factor(formulation)
            h_bonding = self._get_hydrogen_bonding_factor(formulation)
            
            base_value = 2.0 + polarity * 25
            hbond_effect = h_bonding * 15
            final_value = max(2.0, min(80.0, base_value + hbond_effect))
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in dielectric constant prediction: {e}")
            return 10.0, 0.3

    def predict_solubility_parameter(self, formulation: Dict) -> Tuple[float, float]:
        """Predict Hansen solubility parameter in MPa^1/2"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity = self._get_polarity_factor(formulation)
            h_bonding = self._get_hydrogen_bonding_factor(formulation)
            
            base_value = 16 + (avg_mw - 100) / 120
            polarity_effect = polarity * 4
            hbond_effect = h_bonding * 3
            final_value = max(12, min(30, base_value + polarity_effect + hbond_effect))
            
            confidence = 0.65
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in solubility parameter prediction: {e}")
            return 20.0, 0.3

    def predict_absorption_capacity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict absorption capacity"""
        try:
            porosity = self._get_porosity_factor(formulation)
            polarity = self._get_polarity_factor(formulation)
            
            # Porous and polar materials absorb better
            base_value = 0.2 + porosity * 0.6
            polarity_boost = polarity * 0.2
            final_value = min(1.0, base_value + polarity_boost)
            
            confidence = 0.6
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in absorption capacity prediction: {e}")
            return 0.5, 0.3

    def predict_thermal_stability(self, formulation: Dict) -> Tuple[float, float]:
        """Predict thermal stability"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            complexity = self._get_complexity_factor(formulation)
            
            # Larger, more complex molecules are generally more stable
            stability = 0.5 + (avg_mw / 2000) + (complexity * 0.3)
            final_value = min(1.0, stability)
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in thermal stability prediction: {e}")
            return 0.7, 0.3

    def predict_toxicity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict toxicity (lower = safer)"""
        try:
            complexity = self._get_complexity_factor(formulation)
            # Simple heuristic - more complex might be more toxic
            toxicity = complexity * 0.25
            final_value = min(1.0, toxicity)
            
            confidence = 0.4  # Low confidence for toxicity
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in toxicity prediction: {e}")
            return 0.3, 0.2

    # Enhanced helper methods
    def _get_average_molecular_weight(self, formulation: Dict) -> float:
        """Calculate average molecular weight"""
        try:
            total_mass = 0
            weighted_mw = 0
            
            for component in formulation.get('composition', []):
                mass_frac = component.get('mass_percentage', 0) / 100
                mw = self._get_mw(component)
                weighted_mw += mass_frac * mw
                total_mass += mass_frac
                
            return weighted_mw / total_mass if total_mass > 0 else 100
        except Exception as e:
            self.logger.error(f"Error calculating average MW: {e}")
            return 100.0

    def _get_mw(self, compound: Dict) -> float:
        """Safely get molecular weight"""
        try:
            mw = compound.get('molecular_weight')
            if mw is None:
                return 100.0
            if isinstance(mw, str):
                mw = float(mw.strip()) if mw.strip() else 100.0
            if not (0.1 <= mw <= 100000):
                return 100.0
            return float(mw)
        except:
            return 100.0

    def _get_complexity_factor(self, formulation: Dict) -> float:
        """Calculate molecular complexity factor"""
        try:
            complexities = []
            for component in formulation.get('composition', []):
                smiles = component.get('smiles', '')
                name = component.get('name', '').lower()
                
                # Estimate complexity from various factors
                comp_score = 0.5  # Base
                
                # Longer SMILES = more complex
                if smiles:
                    comp_score = min(len(smiles) / 80, 1.0)
                
                # Specific complexity indicators
                if any(indicator in name for indicator in ['polymer', 'complex', 'macro']):
                    comp_score = max(comp_score, 0.8)
                if 'nano' in name:
                    comp_score = max(comp_score, 0.7)
                    
                complexities.append(comp_score)
            
            return np.mean(complexities) if complexities else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating complexity factor: {e}")
            return 0.5

    def _get_polarity_factor(self, formulation: Dict) -> float:
        """Calculate polarity factor"""
        try:
            polarity_scores = []
            for component in formulation.get('composition', []):
                smiles = component.get('smiles', '').lower()
                name = component.get('name', '').lower()
                score = 0.0
                
                # Detect polar functional groups from SMILES and names
                polar_indicators = [
                    ('oh', 0.7), ('cooh', 0.8), ('c=o', 0.6), 
                    ('o', 0.3), ('n', 0.4), ('s=o', 0.5),
                    ('cn', 0.4), ('cl', 0.2), ('f', 0.3),
                    ('alcohol', 0.6), ('acid', 0.7), ('amine', 0.5)
                ]
                
                for indicator, indicator_score in polar_indicators:
                    if indicator in smiles or indicator in name:
                        score = max(score, indicator_score)
                
                polarity_scores.append(score)
            
            return np.mean(polarity_scores) if polarity_scores else 0.3
        except Exception as e:
            self.logger.error(f"Error calculating polarity factor: {e}")
            return 0.3

    def _get_hydrogen_bonding_factor(self, formulation: Dict) -> float:
        """Calculate hydrogen bonding capability"""
        try:
            hbond_scores = []
            for component in formulation.get('composition', []):
                smiles = component.get('smiles', '').lower()
                name = component.get('name', '').lower()
                score = 0.0
                
                # Groups that can hydrogen bond
                hbond_indicators = [
                    ('oh', 0.8), ('cooh', 0.9), ('nh2', 0.7),
                    ('n-h', 0.6), ('o-h', 0.8), ('alcohol', 0.7),
                    ('acid', 0.8), ('amine', 0.6), ('water', 1.0)
                ]
                
                for indicator, indicator_score in hbond_indicators:
                    if indicator in smiles or indicator in name:
                        score = max(score, indicator_score)
                
                hbond_scores.append(score)
            
            return np.mean(hbond_scores) if hbond_scores else 0.3
        except Exception as e:
            self.logger.error(f"Error calculating hydrogen bonding factor: {e}")
            return 0.3

    def _get_porosity_factor(self, formulation: Dict) -> float:
        """Calculate porosity factor"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            complexity = self._get_complexity_factor(formulation)
            
            # Lower MW and higher complexity might indicate porosity
            porosity = (1 - min(avg_mw / 2000, 1)) * complexity
            return min(porosity, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating porosity factor: {e}")
            return 0.5
