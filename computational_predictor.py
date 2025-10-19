import numpy as np
from typing import Dict, List, Any, Tuple
import re
import logging
import random

class ComputationalPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.property_models = {
            'band_gap': self.predict_band_gap,
            'electron_mobility': self.predict_electron_mobility,
            'hole_mobility': self.predict_hole_mobility,
            'absorption_spectrum': self.predict_absorption,
            'quantum_efficiency': self.predict_quantum_efficiency,
            'charge_transfer': self.predict_charge_transfer,
            'exciton_binding': self.predict_exciton_binding,
            'photo_stability': self.predict_photo_stability
        }
        
    def predict_advanced_properties(self, formulation: Dict, target_properties: Dict) -> Dict[str, Any]:
        """Predict advanced electronic and optical properties using AI-enhanced methods"""
        advanced_props = {}
        
        for prop_name in target_properties.keys():
            if prop_name in self.property_models:
                try:
                    value, confidence, reasoning = self.property_models[prop_name](formulation)
                    advanced_props[prop_name] = {
                        'value': value,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'units': self._get_units(prop_name)
                    }
                except Exception as e:
                    self.logger.error(f"Error predicting {prop_name}: {e}")
                    advanced_props[prop_name] = {
                        'value': 0.0,
                        'confidence': 0.1,
                        'reasoning': f"Prediction failed: {str(e)}",
                        'units': self._get_units(prop_name)
                    }
        
        return advanced_props

    def predict_band_gap(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict electronic band gap using AI-enhanced molecular analysis"""
        try:
            compounds = formulation.get('composition', [])
            if not compounds:
                return 2.0, 0.1, "No compounds available"
            
            # AI-enhanced band gap estimation
            band_gaps = []
            reasoning_points = []
            
            for comp in compounds:
                name = comp.get('name', '').lower()
                smiles = comp.get('smiles', '').lower()
                mw = comp.get('molecular_weight', 100)
                
                # Enhanced heuristics based on molecular characteristics
                gap = self._estimate_band_gap_from_properties(name, smiles, mw)
                band_gaps.append(gap)
                reasoning_points.append(f"{comp.get('name')}: {self._get_band_gap_reasoning(name, smiles)}")
            
            if band_gaps:
                # Weighted average considering molecular complexity
                weights = [self._get_complexity_weight(comp) for comp in compounds]
                avg_gap = np.average(band_gaps, weights=weights)
                confidence = min(0.8, len(band_gaps) * 0.15)
                reasoning = f"AI-estimated from molecular properties. Analysis: {', '.join(reasoning_points)}"
                return avg_gap, confidence, reasoning
            else:
                return 2.5, 0.3, "Using default semiconductor band gap"
                
        except Exception as e:
            self.logger.error(f"Band gap prediction error: {e}")
            return 2.5, 0.1, f"Band gap prediction error: {str(e)}"

    def predict_electron_mobility(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict electron mobility using advanced heuristics"""
        try:
            compounds = formulation.get('composition', [])
            mobility_factors = []
            reasoning_points = []
            
            for comp in compounds:
                name = comp.get('name', '').lower()
                smiles = comp.get('smiles', '').lower()
                
                # Enhanced mobility estimation
                mobility_score = self._estimate_mobility(name, smiles)
                mobility_factors.append(mobility_score)
                reasoning_points.append(f"{comp.get('name')}: mobility_score={mobility_score:.1f}")
            
            if mobility_factors:
                avg_mobility = np.mean(mobility_factors)
                confidence = min(0.7, len(mobility_factors) * 0.12)
                reasoning = f"Based on molecular structure and conjugation analysis. Analysis: {', '.join(reasoning_points)}"
                return avg_mobility, confidence, reasoning
            else:
                return 5.0, 0.2, "Default organic semiconductor mobility"
                
        except Exception as e:
            self.logger.error(f"Mobility prediction error: {e}")
            return 5.0, 0.1, f"Mobility prediction error: {str(e)}"

    def predict_absorption(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict absorption characteristics using molecular analysis"""
        try:
            compounds = formulation.get('composition', [])
            absorption_strengths = []
            
            for comp in compounds:
                name = comp.get('name', '').lower()
                smiles = comp.get('smiles', '').lower()
                
                absorption_score = self._estimate_absorption(name, smiles)
                absorption_strengths.append(absorption_score)
            
            if absorption_strengths:
                avg_absorption = np.mean(absorption_strengths)
                confidence = min(0.6, len(absorption_strengths) * 0.1)
                reasoning = "Estimated from chromophore characteristics and molecular conjugation"
                return avg_absorption, confidence, reasoning
            else:
                return 30.0, 0.2, "Default absorption for organic materials"
                
        except Exception as e:
            self.logger.error(f"Absorption prediction error: {e}")
            return 30.0, 0.1, f"Absorption prediction error: {str(e)}"

    def predict_quantum_efficiency(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict quantum efficiency using molecular stability analysis"""
        try:
            compounds = formulation.get('composition', [])
            efficiency_factors = []
            
            for comp in compounds:
                name = comp.get('name', '').lower()
                smiles = comp.get('smiles', '').lower()
                complexity = comp.get('complexity', 50)
                
                efficiency = self._estimate_quantum_efficiency(name, smiles, complexity)
                efficiency_factors.append(efficiency)
            
            if efficiency_factors:
                avg_efficiency = np.mean(efficiency_factors)
                confidence = min(0.5, len(efficiency_factors) * 0.08)
                reasoning = "Based on molecular rigidity and electronic stability analysis"
                return avg_efficiency, confidence, reasoning
            else:
                return 25.0, 0.15, "Default quantum efficiency estimate"
                
        except Exception as e:
            self.logger.error(f"Quantum efficiency prediction error: {e}")
            return 25.0, 0.1, f"Quantum efficiency prediction error: {str(e)}"

    # Enhanced helper methods without RDKit
    def _estimate_band_gap_from_properties(self, name: str, smiles: str, mw: float) -> float:
        """Estimate band gap from molecular properties"""
        base_gap = 3.0
        
        # Conjugated systems have smaller band gaps
        if any(indicator in name or indicator in smiles for indicator in 
               ['conjugated', 'aromatic', 'benzene', 'phenyl', 'naphthalene']):
            base_gap -= 1.2
        
        # Large molecules tend to have smaller band gaps
        if mw > 300:
            base_gap -= 0.5
        elif mw > 500:
            base_gap -= 0.8
            
        # Specific functional groups affect band gap
        if 'c=c' in smiles or 'C=C' in smiles:
            base_gap -= 0.3
        if 'c#c' in smiles or 'C#C' in smiles:
            base_gap -= 0.4
        if 'n' in smiles.lower() and 'c' in smiles.lower():
            base_gap -= 0.2
            
        return max(0.1, min(6.0, base_gap))

    def _estimate_mobility(self, name: str, smiles: str) -> float:
        """Estimate charge carrier mobility"""
        mobility = 5.0  # Base mobility
        
        # Planar conjugated systems have higher mobility
        if any(indicator in name for indicator in ['graphite', 'graphene', 'nanotube', 'fullerene']):
            mobility += 20.0
        elif any(indicator in name or indicator in smiles for indicator in 
                ['conjugated', 'aromatic', 'polymer']):
            mobility += 10.0
            
        # Heteroatoms can facilitate charge transport
        heteroatom_count = sum(1 for atom in ['n', 'o', 's'] if atom in smiles.lower())
        mobility += heteroatom_count * 2.0
        
        return min(100.0, mobility)

    def _estimate_absorption(self, name: str, smiles: str) -> float:
        """Estimate absorption strength"""
        absorption = 30.0  # Base absorption
        
        # Chromophore-rich compounds have higher absorption
        if any(indicator in name for indicator in ['dye', 'chromophore', 'pigment', 'color']):
            absorption += 40.0
            
        # Conjugated systems
        conjugation_indicators = ['c=c-c=c', 'c1ccccc1', 'conjugated', 'aromatic']
        if any(indicator in smiles for indicator in conjugation_indicators):
            absorption += 25.0
            
        # Carbonyl groups enhance absorption
        if 'c=o' in smiles.lower():
            absorption += 15.0
            
        return min(100.0, absorption)

    def _estimate_quantum_efficiency(self, name: str, smiles: str, complexity: float) -> float:
        """Estimate quantum efficiency"""
        efficiency = 25.0  # Base efficiency
        
        # Rigid molecules have better quantum efficiency
        if complexity > 200:
            efficiency += 20.0
        elif complexity > 100:
            efficiency += 10.0
            
        # Conjugated systems
        if any(indicator in name or indicator in smiles for indicator in 
              ['rigid', 'stiff', 'conjugated', 'aromatic']):
            efficiency += 15.0
            
        return min(95.0, efficiency)

    def _get_band_gap_reasoning(self, name: str, smiles: str) -> str:
        """Generate reasoning for band gap estimation"""
        factors = []
        
        if any(indicator in name or indicator in smiles for indicator in ['conjugated', 'aromatic']):
            factors.append("conjugated system")
        if 'c=c' in smiles.lower():
            factors.append("double bonds")
        if 'c#c' in smiles.lower():
            factors.append("triple bonds")
            
        return " + ".join(factors) if factors else "standard molecular structure"

    def _get_complexity_weight(self, compound: Dict) -> float:
        """Get weight based on molecular complexity"""
        complexity = compound.get('complexity', 50)
        return min(2.0, complexity / 100)

    def predict_charge_transfer(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict charge transfer efficiency"""
        try:
            # Enhanced prediction based on formulation properties
            compounds = formulation.get('composition', [])
            if compounds:
                avg_complexity = np.mean([c.get('complexity', 50) for c in compounds])
                efficiency = 0.5 + (avg_complexity / 500)  # More complex = better charge transfer
                efficiency = min(0.9, efficiency)
            else:
                efficiency = 0.7
                
            confidence = 0.4
            reasoning = "Estimated charge transfer based on molecular complexity and electronic structure"
            return efficiency, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Charge transfer prediction error: {e}")
            return 0.7, 0.1, "Default charge transfer estimate"

    def predict_exciton_binding(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict exciton binding energy"""
        try:
            binding_energy = 0.3  # Base value
            compounds = formulation.get('composition', [])
            
            # Adjust based on molecular properties
            for comp in compounds:
                name = comp.get('name', '').lower()
                if 'quantum' in name or 'nano' in name:
                    binding_energy += 0.1
                if 'conjugated' in name:
                    binding_energy -= 0.05
                    
            binding_energy = max(0.1, min(0.5, binding_energy))
            confidence = 0.3
            reasoning = "Estimated exciton binding based on quantum confinement and conjugation"
            return binding_energy, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Exciton binding prediction error: {e}")
            return 0.3, 0.1, "Default exciton binding estimate"

    def predict_photo_stability(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict photostability"""
        try:
            stability = 0.8  # Base stability
            compounds = formulation.get('composition', [])
            
            # More complex molecules tend to be more stable
            if compounds:
                avg_complexity = np.mean([c.get('complexity', 50) for c in compounds])
                stability += (avg_complexity / 1000)  # Slight boost for complexity
                
            stability = min(0.95, stability)
            confidence = 0.5
            reasoning = "Estimated photo-stability based on molecular complexity and bond strength"
            return stability, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Photo-stability prediction error: {e}")
            return 0.8, 0.1, "Default photo-stability estimate"

    def _get_units(self, property_name: str) -> str:
        """Get units for different properties"""
        units_map = {
            'band_gap': 'eV',
            'electron_mobility': 'cm²/V·s',
            'hole_mobility': 'cm²/V·s',
            'absorption_spectrum': 'arbitrary units',
            'quantum_efficiency': '%',
            'charge_transfer': 'eV',
            'exciton_binding': 'eV',
            'photo_stability': 'relative units'
        }
        return units_map.get(property_name, 'unknown')
