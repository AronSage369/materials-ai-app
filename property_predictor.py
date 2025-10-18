# property_predictor.py - ADVANCED PROPERTY PREDICTION
import numpy as np
from typing import Dict, List, Any
import streamlit as st

class AdvancedPropertyPredictor:
    def __init__(self):
        self.prediction_models = {
            'thermal_conductivity': self.predict_thermal_conductivity,
            'viscosity': self.predict_viscosity,
            'flash_point': self.predict_flash_point,
            'specific_heat': self.predict_specific_heat,
            'dielectric_constant': self.predict_dielectric_constant,
            'dielectric_strength': self.predict_dielectric_strength,
            'autoignition_temperature': self.predict_autoignition_temp,
            'pour_point': self.predict_pour_point
        }
        
    def predict_all_properties(self, formulations: List[Dict], strategy: Dict) -> List[Dict]:
        """Predict properties for all formulations with confidence scores"""
        
        target_properties = strategy.get('target_properties', {})
        
        for formulation in formulations:
            compounds = formulation.get('compounds', [])
            ratios = formulation.get('ratios', [1.0] * len(compounds))
            
            if not compounds:
                continue
                
            predicted_properties = {}
            
            # Predict each target property
            for prop_name in target_properties.keys():
                prediction = self.predict_property_for_formulation(compounds, ratios, prop_name)
                predicted_properties[prop_name] = prediction
            
            formulation['predicted_properties'] = predicted_properties
        
        return formulations
    
    def predict_property_for_formulation(self, compounds: List, ratios: List[float], property_name: str) -> Dict[str, Any]:
        """Predict a specific property for a formulation"""
        
        if property_name in self.prediction_models:
            predictor = self.prediction_models[property_name]
            value, confidence, details = predictor(compounds, ratios)
        else:
            # Default prediction
            value, confidence, details = self.default_prediction(compounds, ratios, property_name)
        
        return {
            'value': value,
            'confidence': confidence,
            'unit': self.get_property_unit(property_name),
            'details': details,
            'prediction_method': 'AI_estimation'
        }
    
    def predict_thermal_conductivity(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict thermal conductivity in W/m·K"""
        base_value = 0.08
        confidence = 0.6
        
        # Analyze compound types
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            # Silicones have high thermal conductivity
            if any(term in name for term in ['siloxane', 'silicone']):
                base_value += 0.04 * weight
                confidence += 0.1
            
            # Esters and glycols have moderate thermal conductivity
            elif any(term in name for term in ['ester', 'glycol']):
                base_value += 0.02 * weight
                confidence += 0.05
            
            # Molecular weight effect
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                mw_factor = 1 - abs(compound.molecular_weight - 300) / 1000
                base_value += 0.02 * mw_factor * weight
        
        details = f"Estimated based on {len(compounds)} compounds with silicon/oxygen content analysis"
        return min(0.2, max(0.05, base_value)), min(0.9, confidence), details
    
    def predict_viscosity(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict viscosity in cSt at 100°C"""
        base_viscosity = 0.0
        confidence = 0.7
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                # Base viscosity from molecular weight
                comp_viscosity = compound.molecular_weight / 80
                
                # Adjust for structure
                if any(term in name for term in ['siloxane', 'silicone']):
                    comp_viscosity *= 1.2  # Silicones slightly more viscous
                elif any(term in name for term in ['ester', 'glycol']):
                    comp_viscosity *= 0.9  # Oxygenated compounds less viscous
                elif any(term in name for term in ['oil', 'alkane']):
                    comp_viscosity *= 1.1  # Hydrocarbons more viscous
                
                base_viscosity += comp_viscosity * weight
        
        # Mixing rule for viscosity (logarithmic)
        if base_viscosity > 0:
            log_viscosity = 0
            for i, compound in enumerate(compounds):
                comp_viscosity = base_viscosity  # Simplified
                log_viscosity += ratios[i] * np.log(max(1.0, comp_viscosity))
            final_viscosity = np.exp(log_viscosity)
        else:
            final_viscosity = 10.0  # Default
        
        details = f"Viscosity prediction using molecular weight and structural analysis"
        return min(100.0, max(1.0, final_viscosity)), confidence, details
    
    def predict_flash_point(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict flash point in °C"""
        flash_points = []
        confidence = 0.65
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                base_fp = 50 + (compound.molecular_weight * 0.3)
                
                # Structural adjustments
                if any(term in name for term in ['siloxane', 'silicone']):
                    base_fp += 40  # Silicones have high flash points
                elif any(term in name for term in ['ester', 'glycol']):
                    base_fp += 20  # Oxygenated compounds have higher flash points
                
                flash_points.append(base_fp * weight)
            else:
                flash_points.append(150.0 * weight)  # Default
        
        avg_flash_point = sum(flash_points)
        details = f"Flash point estimation based on molecular weight and functional groups"
        return min(300.0, max(50.0, avg_flash_point)), confidence, details
    
    def predict_specific_heat(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict specific heat in J/kg·K"""
        specific_heat = 0.0
        confidence = 0.6
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            formula = getattr(compound, 'molecular_formula', '')
            
            base_sh = 1800  # Base for organic compounds
            
            # Oxygenated compounds have higher specific heat
            if 'O' in formula:
                base_sh += 300 * formula.count('O')
            
            if any(term in name for term in ['glycol', 'alcohol']):
                base_sh += 400
            elif any(term in name for term in ['ester', 'ether']):
                base_sh += 200
            
            # Molecular weight effect (lighter = higher specific heat)
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                mw_factor = max(0.5, 1000 / compound.molecular_weight)
                base_sh *= mw_factor
            
            specific_heat += base_sh * weight
        
        details = "Specific heat prediction based on oxygen content and molecular structure"
        return min(3000, max(1000, specific_heat)), confidence, details
    
    def predict_dielectric_constant(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict dielectric constant"""
        dielectric_const = 0.0
        confidence = 0.7
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            formula = getattr(compound, 'molecular_formula', '')
            
            base_dc = 2.0  # Base for non-polar
            
            # Polar compounds have higher dielectric constant
            if 'O' in formula or 'N' in formula:
                base_dc += 1.0
            
            if any(term in name for term in ['glycol', 'alcohol']):
                base_dc += 2.0  # Highly polar
            elif any(term in name for term in ['ester', 'ether']):
                base_dc += 1.5
            elif any(term in name for term in ['siloxane', 'silicone']):
                base_dc += 0.5  # Silicones have low dielectric constant
            
            dielectric_const += base_dc * weight
        
        details = "Dielectric constant estimation based on polarity and functional groups"
        return min(10.0, max(1.5, dielectric_const)), confidence, details
    
    def predict_dielectric_strength(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict dielectric strength in kV/2.5mm"""
        dielectric_strength = 0.0
        confidence = 0.65
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            formula = getattr(compound, 'molecular_formula', '')
            
            base_ds = 30.0  # Base value
            
            # Hydrocarbons and silicones have good dielectric strength
            if 'O' not in formula and 'N' not in formula:  # Pure hydrocarbons
                base_ds += 15.0
            elif any(term in name for term in ['siloxane', 'silicone']):
                base_ds += 10.0
            elif any(term in name for term in ['ester', 'oil']):
                base_ds += 5.0
            
            dielectric_strength += base_ds * weight
        
        details = "Dielectric strength prediction based on chemical structure and purity"
        return min(60.0, max(20.0, dielectric_strength)), confidence, details
    
    def predict_autoignition_temp(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict autoignition temperature in °C"""
        autoignition_temp = 0.0
        confidence = 0.6
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                base_ait = 200 + (compound.molecular_weight * 0.4)
                
                # Silicones have very high autoignition temperatures
                if any(term in name for term in ['siloxane', 'silicone']):
                    base_ait += 100
                # Oxygenated compounds have moderate autoignition temperatures
                elif any(term in name for term in ['ester', 'glycol']):
                    base_ait += 50
                
                autoignition_temp += base_ait * weight
            else:
                autoignition_temp += 300.0 * weight  # Default
        
        details = "Autoignition temperature estimation based on molecular weight and stability"
        return min(500.0, max(200.0, autoignition_temp)), confidence, details
    
    def predict_pour_point(self, compounds: List, ratios: List[float]) -> tuple:
        """Predict pour point in °C"""
        pour_point = 0.0
        confidence = 0.7
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            name = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                base_pp = 0 - (compound.molecular_weight * 0.1)  # Higher MW = higher pour point
                
                # Structural effects
                if any(term in name for term in ['branched', 'iso']):
                    base_pp -= 20  # Branched molecules have lower pour points
                elif any(term in name for term in ['linear', 'normal']):
                    base_pp += 10  # Linear molecules have higher pour points
                
                pour_point += base_pp * weight
            else:
                pour_point -= 20.0 * weight  # Default
        
        details = "Pour point prediction based on molecular structure and branching"
        return min(20.0, max(-60.0, pour_point)), confidence, details
    
    def default_prediction(self, compounds: List, ratios: List[float], property_name: str) -> tuple:
        """Default prediction for unknown properties"""
        # Simple average based on compound count and complexity
        base_value = 0.5
        confidence = 0.4
        
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                complexity = min(1.0, compound.molecular_weight / 1000)
                base_value += complexity * weight * 0.3
        
        details = f"Generic prediction for {property_name} based on molecular complexity"
        return min(1.0, max(0.1, base_value)), confidence, details
    
    def get_property_unit(self, property_name: str) -> str:
        """Get unit for property"""
        units = {
            'thermal_conductivity': 'W/m·K',
            'viscosity': 'cSt',
            'flash_point': '°C',
            'specific_heat': 'J/kg·K',
            'dielectric_constant': '',
            'dielectric_strength': 'kV/2.5mm',
            'autoignition_temperature': '°C',
            'pour_point': '°C',
            'cost': '$/kg'
        }
        return units.get(property_name, '')
