import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
import random
import logging
from utils import cached, MemoryManager, PropertyCalculator

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
        
    @cached
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
                        self.logger.error(f"Error predicting {prop_name}: {e}")
                        predicted_properties[prop_name] = 0.0
                        confidence_scores[prop_name] = 0.0
            
            # Enhance predictions for solute-solvent systems
            if any(key in formulation for key in ['solvent_components', 'solute_components']):
                enhanced_properties = self._enhance_solute_solvent_predictions(formulation, predicted_properties)
                formulation['predicted_properties'] = enhanced_properties
            else:
                formulation['predicted_properties'] = predicted_properties
                
            formulation['prediction_confidence'] = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
            
        return formulations

    def predict_mixture_properties(self, formulation: Dict, target_properties: Dict) -> Dict:
        """Predict properties considering solute-solvent interactions"""
        base_properties = self.predict_all_properties([formulation], target_properties)[0]['predicted_properties']
        return base_properties

    def _enhance_solute_solvent_predictions(self, formulation: Dict, base_properties: Dict) -> Dict:
        """Enhance property predictions for solute-solvent systems"""
        enhanced = base_properties.copy()
        composition = formulation.get('composition', [])
        
        # Get solvent and solute information
        solvent_cids = set(formulation.get('solvent_components', []))
        solute_cids = set(formulation.get('solute_components', []))
        
        solvents = [comp for comp in composition if comp.get('cid') in solvent_cids]
        solutes = [comp for comp in composition if comp.get('cid') in solute_cids]
        
        if not solvents or not solutes:
            return enhanced
        
        # Calculate solute concentration
        solute_concentration = sum(comp.get('mass_percentage', 0) for comp in solutes) / 100
        
        # Enhance predictions based on solute concentration
        for prop in enhanced:
            if prop == 'viscosity':
                # Solutes generally increase viscosity
                enhancement = 1 + (solute_concentration * 2)
                enhanced[prop] *= enhancement
            
            elif prop == 'thermal_conductivity':
                # Solid solutes can increase thermal conductivity
                if any(self._is_likely_solid(comp) for comp in solutes):
                    enhancement = 1 + (solute_concentration * 0.5)
                    enhanced[prop] *= enhancement
            
            elif prop == 'density':
                # Adjust density based on solute properties
                avg_solute_mw = np.mean([comp.get('molecular_weight', 100) for comp in solutes])
                avg_solvent_mw = np.mean([comp.get('molecular_weight', 100) for comp in solvents])
                
                if avg_solute_mw > avg_solvent_mw:
                    enhancement = 1 + (solute_concentration * 0.3)
                    enhanced[prop] *= enhancement
            
            elif prop == 'surface_tension':
                # Solutes can affect surface tension
                enhancement = 1 + (solute_concentration * random.uniform(-0.2, 0.5))
                enhanced[prop] *= enhancement
        
        return enhanced

    def _is_likely_solid(self, compound: Dict) -> bool:
        """Heuristic to determine if compound is likely solid at room temperature"""
        mw = compound.get('molecular_weight', 0)
        complexity = compound.get('complexity', 0)
        
        # Simple heuristic: higher MW and complexity often indicate solids
        return mw > 200 and complexity > 150

    def predict_thermal_conductivity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict thermal conductivity in W/m·K"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Lower MW generally higher thermal conductivity
            base_value = 0.1 + (500 - avg_mw) / 1000
            base_value = max(0.05, min(0.8, base_value))
            
            # Adjust for composition complexity
            complexity_factor = self._get_complexity_factor(formulation)
            final_value = base_value * (1.0 + complexity_factor * 0.2)
            
            confidence = 0.7
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in thermal conductivity prediction: {e}")
            return 0.2, 0.3

    def predict_viscosity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict viscosity in cP"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Higher MW generally higher viscosity
            base_value = 0.1 + (avg_mw - 50) / 200
            base_value = max(0.1, min(1000, base_value))
            
            # Adjust for molecular complexity
            complexity_factor = self._get_complexity_factor(formulation)
            final_value = base_value * (1.0 + complexity_factor * 0.5)
            
            confidence = 0.8
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in viscosity prediction: {e}")
            return 10.0, 0.3

    def predict_boiling_point(self, formulation: Dict) -> Tuple[float, float]:
        """Predict boiling point in °C"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Boiling point generally increases with MW
            base_value = 50 + (avg_mw - 50) * 2.0
            base_value = max(-100, min(400, base_value))
            
            # Adjust for polarity and hydrogen bonding
            polarity_factor = self._get_polarity_factor(formulation)
            final_value = base_value * (1.0 + polarity_factor * 0.3)
            
            confidence = 0.75
            return final_value, confidence
        except Exception as e:
            self.logger.error(f"Error in boiling point prediction: {e}")
            return 100.0, 0.3

    def predict_flash_point(self, formulation: Dict) -> Tuple[float, float]:
        """Predict flash point in °C"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Flash point generally increases with MW
            base_value = 30 + (avg_mw - 50) * 1.5
            base_value = max(-50, min(300, base_value))
            
            confidence = 0.65
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in flash point prediction: {e}")
            return 50.0, 0.3

    def predict_specific_heat(self, formulation: Dict) -> Tuple[float, float]:
        """Predict specific heat in J/g·K"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Lower MW generally higher specific heat
            base_value = 1.0 + (200 - avg_mw) / 500
            base_value = max(0.5, min(4.0, base_value))
            
            confidence = 0.7
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in specific heat prediction: {e}")
            return 2.0, 0.3

    def predict_density(self, formulation: Dict) -> Tuple[float, float]:
        """Predict density in g/mL"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Density increases with MW but plateaus
            base_value = 0.7 + (avg_mw - 50) / 1000
            base_value = max(0.6, min(1.5, base_value))
            
            confidence = 0.8
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in density prediction: {e}")
            return 1.0, 0.3

    def predict_surface_tension(self, formulation: Dict) -> Tuple[float, float]:
        """Predict surface tension in mN/m"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Surface tension increases with MW
            base_value = 15 + (avg_mw - 50) / 20
            base_value = max(10, min(80, base_value))
            
            confidence = 0.6
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in surface tension prediction: {e}")
            return 30.0, 0.3

    def predict_refractive_index(self, formulation: Dict) -> Tuple[float, float]:
        """Predict refractive index"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            
            # Heuristic: Refractive index slightly increases with MW
            base_value = 1.3 + (avg_mw - 50) / 2000
            base_value = max(1.3, min(1.7, base_value))
            
            confidence = 0.7
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in refractive index prediction: {e}")
            return 1.4, 0.3

    def predict_dielectric_constant(self, formulation: Dict) -> Tuple[float, float]:
        """Predict dielectric constant"""
        try:
            polarity_factor = self._get_polarity_factor(formulation)
            
            # Heuristic: Higher polarity = higher dielectric constant
            base_value = 2.0 + polarity_factor * 30.0
            base_value = max(2.0, min(80.0, base_value))
            
            confidence = 0.65
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in dielectric constant prediction: {e}")
            return 10.0, 0.3

    def predict_solubility_parameter(self, formulation: Dict) -> Tuple[float, float]:
        """Predict Hansen solubility parameter in MPa^1/2"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            polarity_factor = self._get_polarity_factor(formulation)
            
            # Heuristic: Combination of MW and polarity effects
            base_value = 15 + (avg_mw - 100) / 100 + polarity_factor * 5
            base_value = max(10, min(30, base_value))
            
            confidence = 0.6
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in solubility parameter prediction: {e}")
            return 20.0, 0.3

    def predict_absorption_capacity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict absorption capacity (generic)"""
        try:
            # Simple heuristic based on molecular properties
            porosity_factor = self._get_porosity_factor(formulation)
            base_value = 0.1 + porosity_factor * 0.9
            confidence = 0.5
            return base_value, confidence
        except Exception as e:
            self.logger.error(f"Error in absorption capacity prediction: {e}")
            return 0.5, 0.3

    def predict_thermal_stability(self, formulation: Dict) -> Tuple[float, float]:
        """Predict thermal stability (higher = more stable)"""
        try:
            avg_mw = self._get_average_molecular_weight(formulation)
            # Higher MW often correlates with better thermal stability
            stability = min(1.0, avg_mw / 1000)
            confidence = 0.6
            return stability, confidence
        except Exception as e:
            self.logger.error(f"Error in thermal stability prediction: {e}")
            return 0.7, 0.3

    def predict_toxicity(self, formulation: Dict) -> Tuple[float, float]:
        """Predict toxicity (lower = safer)"""
        try:
            # Simple heuristic: complex molecules and certain functional groups might be more toxic
            complexity = self._get_complexity_factor(formulation)
            toxicity = complexity * 0.3  # Basic correlation
            confidence = 0.4  # Low confidence for toxicity predictions
            return toxicity, confidence
        except Exception as e:
            self.logger.error(f"Error in toxicity prediction: {e}")
            return 0.3, 0.2

    def _get_average_molecular_weight(self, formulation: Dict) -> float:
        """Calculate average molecular weight of formulation"""
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
        try:
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
        except Exception as e:
            self.logger.error(f"Error calculating complexity factor: {e}")
            return 0.5

    def _get_polarity_factor(self, formulation: Dict) -> float:
        """Calculate polarity factor based on functional groups"""
        try:
            polarity_scores = []
            
            for component in formulation.get('composition', []):
                smiles = component.get('smiles', '').lower()
                score = 0.0
                
                # Detect polar functional groups
                polar_groups = [
                    ('oh', 0.7),  # Hydroxyl groups
                    ('cooh', 0.8), # Carboxylic acid
                    ('c=o', 0.6), # Carbonyl groups
                    ('o', 0.3),   # Oxygen atoms
                    ('n', 0.4),   # Nitrogen atoms
                    ('s=o', 0.5), # Sulfoxide/sulfone
                    ('cn', 0.4),  # Nitrile
                    ('cl', 0.2),  # Chlorine
                    ('f', 0.3),   # Fluorine
                ]
                
                for group, group_score in polar_groups:
                    if group in smiles:
                        score = max(score, group_score)
                
                polarity_scores.append(score)
            
            return np.mean(polarity_scores) if polarity_scores else 0.3
        except Exception as e:
            self.logger.error(f"Error calculating polarity factor: {e}")
            return 0.3

    def _get_porosity_factor(self, formulation: Dict) -> float:
        """Calculate porosity factor for absorbent materials"""
        try:
            # Simple heuristic based on molecular size and branching
            avg_mw = self._get_average_molecular_weight(formulation)
            complexity = self._get_complexity_factor(formulation)
            
            # Lower MW and higher complexity might indicate more porous structures
            porosity = (1 - min(avg_mw / 2000, 1)) * complexity
            return min(porosity, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating porosity factor: {e}")
            return 0.5
