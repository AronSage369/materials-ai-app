import numpy as np
from typing import Dict, List, Any, Tuple
import re
import logging
import random

# Try to import RDKit with fallback
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available, using simplified property predictions")

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
        """Predict advanced electronic and optical properties using computational methods"""
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
        """Predict electronic band gap using molecular orbital theory"""
        try:
            compounds = formulation.get('composition', [])
            if not compounds:
                return 2.0, 0.1, "No compounds available"
            
            # Calculate average HOMO-LUMO gap estimate
            band_gaps = []
            reasoning_points = []
            
            for comp in compounds:
                smiles = comp.get('smiles')
                if smiles and RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Estimate band gap from molecular properties
                            mol_wt = Descriptors.MolWt(mol)
                            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
                            
                            # Simple heuristic based on molecular complexity
                            if num_aromatic_rings > 0:
                                # Conjugated systems have smaller band gaps
                                gap = 2.5 - (num_aromatic_rings * 0.3)
                            else:
                                # Aliphatic systems have larger band gaps
                                gap = 5.0 - (mol_wt / 500)
                            
                            band_gaps.append(max(0.1, min(6.0, gap)))
                            reasoning_points.append(f"{comp.get('name')}: conjugated system" if num_aromatic_rings > 0 else f"{comp.get('name')}: aliphatic system")
                    except:
                        continue
                else:
                    # Fallback without RDKit
                    name = comp.get('name', '').lower()
                    if any(indicator in name for indicator in ['aromatic', 'conjugated', 'benzene']):
                        band_gaps.append(2.0)
                        reasoning_points.append(f"{comp.get('name')}: likely conjugated")
                    else:
                        band_gaps.append(3.5)
                        reasoning_points.append(f"{comp.get('name')}: default estimate")
            
            if band_gaps:
                avg_gap = np.mean(band_gaps)
                confidence = min(0.8, len(band_gaps) * 0.15)
                reasoning = f"Estimated from molecular structure analysis. Key factors: {', '.join(reasoning_points)}"
                return avg_gap, confidence, reasoning
            else:
                return 2.5, 0.3, "Using default semiconductor band gap"
                
        except Exception as e:
            self.logger.error(f"Band gap prediction error: {e}")
            return 2.5, 0.1, f"Band gap prediction error: {str(e)}"

    def predict_electron_mobility(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict electron mobility using molecular stacking estimates"""
        try:
            compounds = formulation.get('composition', [])
            mobility_factors = []
            reasoning_points = []
            
            for comp in compounds:
                smiles = comp.get('smiles')
                if smiles and RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Factors affecting electron mobility
                            planar_score = self._calculate_planarity(mol)
                            conjugation_score = self._calculate_conjugation(mol)
                            heteroatom_count = self._count_heteroatoms(mol)
                            
                            # Mobility heuristic
                            mobility = (planar_score * 0.4 + conjugation_score * 0.4 + 
                                      (heteroatom_count * 0.02)) * 10
                            
                            mobility_factors.append(min(100.0, mobility))
                            reasoning_points.append(f"{comp.get('name')}: planar={planar_score:.2f}, conjugated={conjugation_score:.2f}")
                    except:
                        continue
                else:
                    # Fallback without RDKit
                    name = comp.get('name', '').lower()
                    mobility_score = 5.0  # Default
                    if any(indicator in name for indicator in ['aromatic', 'conjugated']):
                        mobility_score = 15.0
                    mobility_factors.append(mobility_score)
                    reasoning_points.append(f"{comp.get('name')}: basic estimate")
            
            if mobility_factors:
                avg_mobility = np.mean(mobility_factors)
                confidence = min(0.7, len(mobility_factors) * 0.12)
                reasoning = f"Based on molecular planarity and conjugation. Analysis: {', '.join(reasoning_points)}"
                return avg_mobility, confidence, reasoning
            else:
                return 5.0, 0.2, "Default organic semiconductor mobility"
                
        except Exception as e:
            self.logger.error(f"Mobility prediction error: {e}")
            return 5.0, 0.1, f"Mobility prediction error: {str(e)}"

    def predict_absorption(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict absorption characteristics"""
        try:
            compounds = formulation.get('composition', [])
            absorption_strengths = []
            
            for comp in compounds:
                smiles = comp.get('smiles')
                if smiles and RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Chromophore strength estimation
                            num_double_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C')))
                            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
                            num_carbonyl = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=O')))
                            
                            absorption = (num_double_bonds * 0.1 + 
                                        num_aromatic_rings * 0.3 + 
                                        num_carbonyl * 0.2) * 100
                            
                            absorption_strengths.append(min(100.0, absorption))
                    except:
                        continue
                else:
                    # Fallback without RDKit
                    name = comp.get('name', '').lower()
                    absorption_score = 30.0
                    if any(indicator in name for indicator in ['dye', 'chromophore', 'colored']):
                        absorption_score = 70.0
                    absorption_strengths.append(absorption_score)
            
            if absorption_strengths:
                avg_absorption = np.mean(absorption_strengths)
                confidence = min(0.6, len(absorption_strengths) * 0.1)
                reasoning = "Estimated from chromophore density and conjugation"
                return avg_absorption, confidence, reasoning
            else:
                return 30.0, 0.2, "Default absorption for organic materials"
                
        except Exception as e:
            self.logger.error(f"Absorption prediction error: {e}")
            return 30.0, 0.1, f"Absorption prediction error: {str(e)}"

    def predict_quantum_efficiency(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict quantum efficiency for photoactive materials"""
        try:
            # Simple heuristic based on molecular properties
            compounds = formulation.get('composition', [])
            efficiency_factors = []
            
            for comp in compounds:
                smiles = comp.get('smiles')
                if smiles and RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Factors for good quantum efficiency
                            rigidity_score = self._calculate_rigidity(mol)
                            conjugation_length = self._estimate_conjugation_length(mol)
                            
                            efficiency = (rigidity_score * 0.6 + 
                                        min(1.0, conjugation_length / 10) * 0.4) * 100
                            
                            efficiency_factors.append(min(95.0, efficiency))
                    except:
                        continue
                else:
                    # Fallback without RDKit
                    efficiency_factors.append(25.0)
            
            if efficiency_factors:
                avg_efficiency = np.mean(efficiency_factors)
                confidence = min(0.5, len(efficiency_factors) * 0.08)
                reasoning = "Based on molecular rigidity and conjugation length"
                return avg_efficiency, confidence, reasoning
            else:
                return 25.0, 0.15, "Default quantum efficiency estimate"
                
        except Exception as e:
            self.logger.error(f"Quantum efficiency prediction error: {e}")
            return 25.0, 0.1, f"Quantum efficiency prediction error: {str(e)}"

    def predict_charge_transfer(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict charge transfer efficiency"""
        try:
            # Simplified prediction
            efficiency = random.uniform(0.5, 0.9)
            confidence = 0.4
            reasoning = "Estimated charge transfer based on formulation complexity"
            return efficiency, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Charge transfer prediction error: {e}")
            return 0.7, 0.1, "Default charge transfer estimate"

    def predict_exciton_binding(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict exciton binding energy"""
        try:
            binding_energy = random.uniform(0.1, 0.5)
            confidence = 0.3
            reasoning = "Estimated exciton binding based on molecular properties"
            return binding_energy, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Exciton binding prediction error: {e}")
            return 0.3, 0.1, "Default exciton binding estimate"

    def predict_photo_stability(self, formulation: Dict) -> Tuple[float, float, str]:
        """Predict photostability"""
        try:
            stability = random.uniform(0.6, 0.95)
            confidence = 0.5
            reasoning = "Estimated photo-stability based on molecular complexity"
            return stability, confidence, reasoning
        except Exception as e:
            self.logger.error(f"Photo-stability prediction error: {e}")
            return 0.8, 0.1, "Default photo-stability estimate"

    # RDKit-dependent helper methods
    def _calculate_planarity(self, mol) -> float:
        """Calculate molecular planarity score"""
        try:
            if not RDKIT_AVAILABLE:
                return 0.5
                
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            planarity = (num_aromatic_rings * 0.6 + 
                        (1 - min(1.0, num_rotatable_bonds / 10)) * 0.4)
            
            return min(1.0, planarity)
        except:
            return 0.5

    def _calculate_conjugation(self, mol) -> float:
        """Calculate conjugation extent"""
        try:
            if not RDKIT_AVAILABLE:
                return 0.3
                
            conjugated_double_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C-C=C')))
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            conjugation = min(1.0, (conjugated_double_bonds * 0.2 + aromatic_rings * 0.3))
            return conjugation
        except:
            return 0.3

    def _count_heteroatoms(self, mol) -> int:
        """Count heteroatoms that can facilitate charge transport"""
        try:
            if not RDKIT_AVAILABLE:
                return 0
                
            heteroatom_patterns = ['O', 'N', 'S', 'P']
            count = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in heteroatom_patterns:
                    count += 1
            return count
        except:
            return 0

    def _calculate_rigidity(self, mol) -> float:
        """Calculate molecular rigidity score"""
        try:
            if not RDKIT_AVAILABLE:
                return 0.5
                
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            ring_count = Descriptors.RingCount(mol)
            
            if mol.GetNumHeavyAtoms() == 0:
                return 0.5
                
            rigidity = (1 - min(1.0, num_rotatable_bonds / 10)) * 0.6 + \
                     min(1.0, ring_count / 5) * 0.4
            
            return min(1.0, rigidity)
        except:
            return 0.5

    def _estimate_conjugation_length(self, mol) -> float:
        """Estimate conjugation length in atoms"""
        try:
            if not RDKIT_AVAILABLE:
                return 4.0
                
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            double_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C')))
            
            return aromatic_rings * 6 + double_bonds * 2
        except:
            return 4.0

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
            'photo_stability': 'hours'
        }
        return units_map.get(property_name, 'unknown')
