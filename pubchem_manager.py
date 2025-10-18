import pubchempy as pcp
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import random
import time
from dataclasses import dataclass
import numpy as np  # Added missing import

@dataclass
class Compound:
    cid: int
    name: str
    molecular_weight: float
    molecular_formula: str
    smiles: str
    iupac_name: str
    exact_mass: float
    complexity: float

class PubChemManager:
    def __init__(self):
        self.session = None
        self.fallback_compounds = self._initialize_fallback_database()
        self.search_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def find_compounds(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Find compounds using PubChem with fallback options"""
        search_terms = self._generate_search_terms(strategy, material_type)
        
        # Try primary search
        compounds = self._search_pubchem(search_terms, material_type)
        
        # If insufficient results, use fallback
        if len(compounds) < 5:
            fallback_compounds = self._get_fallback_compounds(material_type)
            compounds.extend(fallback_compounds)
        
        # Score and categorize candidates
        scored_compounds = self.score_and_categorize_candidates(
            compounds, strategy.get('target_properties', {})
        )
        
        return scored_compounds

    def _generate_search_terms(self, strategy: Dict, material_type: str) -> List[str]:
        """Generate intelligent search terms based on strategy"""
        base_terms = {
            'solvent': ['polar solvent', 'non-polar solvent', 'aprotic solvent', 'green solvent'],
            'coolant': ['heat transfer fluid', 'thermal fluid', 'dielectric coolant', 'refrigerant'],
            'absorbent': ['absorbent material', 'porous material', 'adsorbent', 'molecular sieve'],
            'catalyst': ['catalyst', 'enzyme', 'zeolite', 'metal complex'],
            'polymer': ['polymer', 'copolymer', 'resin', 'elastomer']
        }
        
        terms = base_terms.get(material_type.lower(), [material_type])
        
        # Add property-based terms
        target_props = strategy.get('target_properties', {})
        if 'boiling_point' in target_props:
            terms.extend(['high boiling', 'low boiling'])
        if 'thermal_conductivity' in target_props:
            terms.extend(['high thermal conductivity', 'heat conductor'])
        if 'viscosity' in target_props:
            terms.extend(['low viscosity', 'high viscosity'])
            
        return terms

    def _search_pubchem(self, search_terms: List[str], material_type: str, 
                       max_compounds: int = 50) -> List[Dict]:
        """Search PubChem for compounds"""
        all_compounds = []
        
        for term in search_terms[:5]:  # Limit to 5 terms to avoid rate limiting
            try:
                print(f"Searching PubChem for: {term}")
                compounds = pcp.get_compounds(term, 'name', listkey_count=10)
                
                for compound in compounds:
                    try:
                        comp_data = self._extract_compound_data(compound)
                        if comp_data and self._filter_compound(comp_data, material_type):
                            all_compounds.append(comp_data)
                    except Exception as e:
                        print(f"Error processing compound {compound.cid}: {e}")
                        continue
                        
                # Brief delay to be respectful to PubChem
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error searching for '{term}': {e}")
                continue
        
        # Remove duplicates and limit results
        unique_compounds = self._remove_duplicate_compounds(all_compounds)
        return unique_compounds[:max_compounds]

    def _extract_compound_data(self, compound) -> Optional[Dict]:
        """Extract relevant data from PubChem compound"""
        try:
            # Get basic properties
            properties = ['MolecularWeight', 'MolecularFormula', 'CanonicalSMILES', 
                         'IUPACName', 'XLogP', 'Complexity']
            
            comp_props = pcp.get_properties(properties, compound.cid)[0] if compound.cid else {}
            
            return {
                'cid': compound.cid,
                'name': compound.synonyms[0] if compound.synonyms else compound.iupac_name,
                'molecular_weight': comp_props.get('MolecularWeight'),
                'molecular_formula': comp_props.get('MolecularFormula'),
                'smiles': comp_props.get('CanonicalSMILES'),
                'iupac_name': comp_props.get('IUPACName'),
                'logp': comp_props.get('XLogP'),
                'complexity': comp_props.get('Complexity'),
                'synonyms': compound.synonyms[:5] if compound.synonyms else []
            }
        except Exception as e:
            print(f"Error extracting data for compound: {e}")
            return None

    def _filter_compound(self, compound: Dict, material_type: str) -> bool:
        """Filter compounds based on material type and basic criteria"""
        # Basic sanity checks
        if not compound.get('molecular_weight') or compound['molecular_weight'] <= 0:
            return False
            
        if not compound.get('smiles'):
            return False
            
        # Material-specific filtering
        mw = compound['molecular_weight']
        
        if material_type.lower() == 'solvent':
            return 30 <= mw <= 500  # Reasonable solvent MW range
        elif material_type.lower() == 'polymer':
            return mw >= 1000  # Polymers typically have higher MW
        else:
            return 50 <= mw <= 2000  # General range for other materials

    def _remove_duplicate_compounds(self, compounds: List[Dict]) -> List[Dict]:
        """Remove duplicate compounds based on CID"""
        seen_cids = set()
        unique_compounds = []
        
        for compound in compounds:
            if compound['cid'] not in seen_cids:
                seen_cids.add(compound['cid'])
                unique_compounds.append(compound)
                
        return unique_compounds

    def score_and_categorize_candidates(self, candidates: List[Dict], 
                                      target_properties: Dict) -> List[Dict]:
        """Score and categorize compounds based on target properties"""
        scored_candidates = []
        
        for candidate in candidates:
            candidate['scores'] = {}
            total_score = 0
            property_count = 0
            
            # Score based on molecular weight heuristic
            mw_score = self._score_by_molecular_weight(candidate, target_properties)
            if mw_score > 0:
                candidate['scores']['molecular_weight'] = mw_score
                total_score += mw_score
                property_count += 1
            
            # Score based on complexity
            complexity_score = self._score_by_complexity(candidate, target_properties)
            if complexity_score > 0:
                candidate['scores']['complexity'] = complexity_score
                total_score += complexity_score
                property_count += 1
                
            # Calculate overall score
            candidate['overall_score'] = total_score / property_count if property_count > 0 else 0.5
            
            # Categorize based on score pattern
            candidate['category'] = self._categorize_compound(candidate, target_properties)
            scored_candidates.append(candidate)
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        return scored_candidates

    def _score_by_molecular_weight(self, candidate: Dict, target_properties: Dict) -> float:
        """Score compound based on molecular weight heuristics"""
        mw = candidate.get('molecular_weight', 0)
        if not mw or mw <= 0:
            return 0.5
            
        # Property-specific MW optima
        if any(prop in target_properties for prop in ['viscosity', 'diffusion_coefficient']):
            # Lower MW generally better for low viscosity/fast diffusion
            return max(0, 1 - (mw - 100) / 1000)
        elif any(prop in target_properties for prop in ['thermal_stability', 'boiling_point']):
            # Higher MW generally better for thermal properties
            return min(1, mw / 500)
        else:
            # Balanced approach
            return 1 - abs(mw - 200) / 400  # Peak around 200 g/mol

    def _score_by_complexity(self, candidate: Dict, target_properties: Dict) -> float:
        """Score compound based on molecular complexity"""
        complexity = candidate.get('complexity', 0)
        if not complexity or complexity <= 0:
            return 0.5
            
        # Normalize complexity score
        normalized_complexity = min(complexity / 500, 1.0)
        
        # Simpler molecules often preferred for solvents/coolants
        if any(prop in target_properties for prop in ['viscosity', 'cost', 'synthesis']):
            return 1 - normalized_complexity * 0.7
        else:
            return 0.5 + normalized_complexity * 0.3

    def _categorize_compound(self, candidate: Dict, target_properties: Dict) -> str:
        """Categorize compound as balanced or specialist"""
        scores = candidate.get('scores', {})
        
        if len(scores) >= 2:
            score_values = list(scores.values())
            variance = np.var(score_values) if score_values else 0
            
            if variance < 0.1:  # Consistent scores across properties
                return 'balanced'
            else:
                return 'specialist'
        else:
            return 'balanced'

    def _initialize_fallback_database(self) -> Dict[str, List[Dict]]:
        """Initialize fallback database with known compounds"""
        return {
            'solvent': [
                {'cid': 887, 'name': 'Methanol', 'molecular_weight': 32.04, 'molecular_formula': 'CH4O', 
                 'smiles': 'CO', 'category': 'balanced', 'iupac_name': 'methanol'},
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'molecular_formula': 'H2O',
                 'smiles': 'O', 'category': 'balanced', 'iupac_name': 'water'},
                {'cid': 6344, 'name': 'Ethanol', 'molecular_weight': 46.07, 'molecular_formula': 'C2H6O',
                 'smiles': 'CCO', 'category': 'balanced', 'iupac_name': 'ethanol'},
                {'cid': 6579, 'name': 'Acetone', 'molecular_weight': 58.08, 'molecular_formula': 'C3H6O',
                 'smiles': 'CC(=O)C', 'category': 'balanced', 'iupac_name': 'propan-2-one'},
                {'cid': 7507, 'name': 'Dimethyl Sulfoxide', 'molecular_weight': 78.13, 'molecular_formula': 'C2H6OS',
                 'smiles': 'CS(C)=O', 'category': 'specialist', 'iupac_name': 'dimethyl sulfoxide'}
            ],
            'coolant': [
                {'cid': 6344, 'name': 'Ethanol', 'molecular_weight': 46.07, 'molecular_formula': 'C2H6O',
                 'smiles': 'CCO', 'category': 'balanced', 'iupac_name': 'ethanol'},
                {'cid': 887, 'name': 'Methanol', 'molecular_weight': 32.04, 'molecular_formula': 'CH4O',
                 'smiles': 'CO', 'category': 'balanced', 'iupac_name': 'methanol'},
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'molecular_formula': 'H2O',
                 'smiles': 'O', 'category': 'balanced', 'iupac_name': 'water'},
                {'cid': 6549, 'name': 'Ethylene Glycol', 'molecular_weight': 62.07, 'molecular_formula': 'C2H6O2',
                 'smiles': 'OCCO', 'category': 'specialist', 'iupac_name': 'ethane-1,2-diol'}
            ],
            'polymer': [
                {'cid': 24756, 'name': 'Polyethylene', 'molecular_weight': 28000, 'molecular_formula': '(C2H4)n',
                 'smiles': 'C=C', 'category': 'balanced', 'iupac_name': 'poly(ethene)'},
                {'cid': 84971, 'name': 'Polystyrene', 'molecular_weight': 10400, 'molecular_formula': '(C8H8)n',
                 'smiles': 'C=Cc1ccccc1', 'category': 'balanced', 'iupac_name': 'poly(1-phenylethane-1,2-diyl)'}
            ]
        }

    def _get_fallback_compounds(self, material_type: str) -> List[Dict]:
        """Get fallback compounds for when PubChem is unavailable"""
        return self.fallback_compounds.get(material_type.lower(), [])
