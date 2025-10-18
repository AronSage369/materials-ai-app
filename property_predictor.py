# property_predictor.py - ADVANCED PROPERTY PREDICTION (FIXED)
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
            'pour_point': self.predict_pour_point,
            'cost': self.predict_cost,
            'surface_area': self.predict_surface_area,
            'co2_capacity': self.predict_co2_capacity,
            'stability': self.predict_stability
        }
        
    def predict_all_properties(self, formulations: List[Dict], strategy: Dict) -> List[Dict]:
        target_properties = strategy.get('target_properties', {})
        for formulation in formulations:
            compounds = formulation.get('compounds', [])
            ratios = formulation.get('ratios', [1.0] * len(compounds))
            if not compounds: continue
            
            predicted_properties = {}
            for prop_name in target_properties.keys():
                prediction = self.predict_property_for_formulation(compounds, ratios, prop_name)
                predicted_properties[prop_name] = prediction
            formulation['predicted_properties'] = predicted_properties
        return formulations
    
    def predict_property_for_formulation(self, compounds: List, ratios: List[float], property_name: str) -> Dict[str, Any]:
        predictor = self.prediction_models.get(property_name, self.default_prediction)
        value, confidence, details = predictor(compounds, ratios)
        return {
            'value': value,
            'confidence': confidence,
            'unit': self.get_property_unit(property_name),
            'details': details,
            'prediction_method': 'heuristic_estimation'
        }

    # --- HELPER FOR SAFE MOLECULAR WEIGHT ACCESS ---
    def _get_mw(self, compound) -> float:
        """Safely gets molecular weight as a float, returning 0.0 on failure."""
        try:
            # FIX: Consistently cast molecular_weight to float for calculations.
            return float(getattr(compound, 'molecular_weight', 0.0))
        except (ValueError, TypeError):
            return 0.0

    # --- PREDICTION MODELS ---
    def predict_thermal_conductivity(self, compounds: List, ratios: List[float]) -> tuple:
        base_value, confidence = 0.0, 0.6
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            mw = self._get_mw(compound)
            if mw == 0.0: continue
            
            comp_val = 0.08 # Base for organic liquids
            mw_factor = 1 - abs(mw - 300) / 1000
            comp_val += mw_factor * 0.05
            base_value += comp_val * weight
            
        details = f"Estimated based on molecular weight and structure of {len(compounds)} compounds."
        return max(0.05, min(0.2, base_value)), confidence, details

    def predict_viscosity(self, compounds: List, ratios: List[float]) -> tuple:
        log_viscosity, confidence = 0.0, 0.7
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            mw = self._get_mw(compound)
            if mw == 0.0: continue
            
            comp_visc = mw / 80.0
            log_viscosity += weight * np.log(max(1.0, comp_visc))
            
        final_viscosity = np.exp(log_viscosity) if log_viscosity else 10.0
        details = "Logarithmic mixing rule applied based on molecular weight."
        return max(1.0, min(100.0, final_viscosity)), confidence, details

    def predict_flash_point(self, compounds: List, ratios: List[float]) -> tuple:
        flash_point, confidence = 0.0, 0.65
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            mw = self._get_mw(compound)
            if mw == 0.0:
                flash_point += 150.0 * weight
                continue
            
            base_fp = 50 + (mw * 0.3)
            flash_point += base_fp * weight
            
        details = "Weighted average based on molecular weight estimation."
        return max(50.0, min(300.0, flash_point)), confidence, details

    def predict_specific_heat(self, compounds: List, ratios: List[float]) -> tuple:
        specific_heat, confidence = 0.0, 0.6
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            mw = self._get_mw(compound)
            base_sh = 1800.0
            if mw > 0:
                mw_factor = max(0.5, 1000 / mw)
                base_sh *= mw_factor
            specific_heat += base_sh * weight
        details = "Estimated based on inverse relationship with molecular weight."
        return max(1000, min(3000, specific_heat)), confidence, details

    def predict_cost(self, compounds: List, ratios: List[float]) -> tuple:
        cost, confidence = 0.0, 0.4
        for i, compound in enumerate(compounds):
            weight = ratios[i]
            mw = self._get_mw(compound)
            if mw == 0.0: continue
            
            # Heuristic: cost increases with complexity (MW) but commodity chems are cheaper
            comp_cost = 5 + (mw / 20)
            if mw < 100: comp_cost *= 0.5 # Cheaper commodity chemicals
            cost += comp_cost * weight
        
        details = "Heuristic cost based on molecular weight as a proxy for complexity."
        return max(1.0, cost), confidence, details
        
    def predict_surface_area(self, compounds: List, ratios: List[float]) -> tuple:
        surface_area, confidence = 0.0, 0.5
        for i, compound in enumerate(compounds):
            name = getattr(compound, 'iupac_name', "").lower()
            if any(term in name for term in ['zeolite', 'mof', 'porous', 'activated']):
                surface_area += (500 + (hash(name) % 1500)) * ratios[i]
                confidence = 0.7
            else:
                surface_area += (5 + (hash(name) % 20)) * ratios[i]
        details = "Estimated based on material type (porous vs. non-porous)."
        return max(1, min(2500, surface_area)), confidence, details

    # Add other predictors here...
    def predict_dielectric_strength(self, compounds: List, ratios: List[float]) -> tuple:
        value, confidence = 0.0, 0.65
        for i, c in enumerate(compounds):
            base_ds = 30.0
            formula = getattr(c, 'molecular_formula', '')
            if 'O' not in formula and 'N' not in formula: base_ds += 15.0
            value += base_ds * ratios[i]
        return max(20.0, min(60.0, value)), confidence, "Based on chemical structure purity."

    def predict_co2_capacity(self, compounds: List, ratios: List[float]) -> tuple:
        value, confidence = 0.0, 0.5
        for i, c in enumerate(compounds):
            mw = self._get_mw(c)
            name = getattr(c, 'iupac_name', "").lower()
            base_cap = 0.5
            if 'amine' in name or 'nh' in name: base_cap += 2.0
            if mw > 0: base_cap += 50 / mw
            value += base_cap * ratios[i]
        return max(0.1, min(5.0, value)), confidence, "Based on amine groups and molecular weight."

    def predict_stability(self, compounds: List, ratios: List[float]) -> tuple:
        value, confidence = 0.0, 0.6
        for i, c in enumerate(compounds):
             mw = self._get_mw(c)
             base_stab = 0.95 - (mw / 5000) # Higher MW, slightly less stable
             value += base_stab * ratios[i]
        return max(0.5, min(0.98, value)), confidence, "Based on molecular weight as a proxy for complexity."


    def default_prediction(self, compounds: List, ratios: List[float], property_name: str) -> tuple:
        base_value, confidence = 0.5, 0.4
        for i, compound in enumerate(compounds):
            mw = self._get_mw(compound)
            if mw == 0.0: continue
            complexity = min(1.0, mw / 1000)
            base_value += complexity * ratios[i] * 0.3
        details = f"Generic prediction for '{property_name}' based on molecular complexity."
        return min(1.0, max(0.1, base_value)), confidence, details
    
    def get_property_unit(self, property_name: str) -> str:
        units = {
            'thermal_conductivity': 'W/m·K', 'viscosity': 'cSt', 'flash_point': '°C',
            'specific_heat': 'J/kg·K', 'dielectric_strength': 'kV/2.5mm', 'cost': '$/kg',
            'surface_area': 'm²/g', 'co2_capacity': 'mmol/g', 'stability': 'score (0-1)'
        }
        return units.get(property_name, '')

