# pubchem_manager.py - CORRECTED INDENTATION
import pubchempy as pcp
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any
import streamlit as st

class PubChemManager:
    def __init__(self):
        self.search_cache = {}
        self.property_cache = {}
        self.rate_limit_delay = 0.3

    def find_compounds(self, strategy: Dict, max_compounds: int, search_depth: str) -> Dict[str, Any]:
        search_params = self._get_search_parameters(search_depth, max_compounds)
        
        search_terms = strategy.get('search_strategy', {}).get('search_terms', [])
        chemical_classes = strategy.get('search_strategy', {}).get('chemical_classes', [])
        
        if not search_terms:
            search_terms = self._generate_default_search_terms(strategy['material_class'])
        
        st.info(f"🔍 Searching for: {', '.join(search_terms[:5])}...")
        
        specialists = self._find_property_specialists(strategy, search_terms, search_params)
        balanced_candidates = self._find_balanced_compounds(strategy, search_terms, search_params)
        
        return {
            'specialists': specialists,
            'balanced': balanced_candidates,
            'search_metrics': {
                'terms_searched': len(search_terms),
                'total_compounds_found': sum(len(s) for s in specialists.values()) + len(balanced_candidates)
            }
        }

    def _get_search_parameters(self, search_depth: str, max_compounds: int) -> Dict:
        params = {
            "Quick Scan": {'compounds_per_term': 5, 'max_terms': 4, 'timeout': 30},
            "Standard Analysis": {'compounds_per_term': 10, 'max_terms': 6, 'timeout': 60},
            "Comprehensive Search": {'compounds_per_term': 15, 'max_terms': 8, 'timeout': 120}
        }
        
        base_params = params.get(search_depth, params["Standard Analysis"])
        base_params['max_compounds'] = max_compounds
        return base_params

    def _generate_default_search_terms(self, material_class: str) -> List[str]:
        term_libraries = {
            'coolant': [
                'dimethyl siloxane', 'polydimethylsiloxane', 'hexamethyldisiloxane',
                'octamethyltrisiloxane', 'decamethylcyclopentasiloxane',
                'mineral oil', 'paraffin oil', 'white oil',
                'propylene glycol', 'ethylene glycol', 'polyethylene glycol',
                'polyalphaolefin', 'synthetic hydrocarbon', 'alkylbenzene',
                'ester oil', 'diester', 'polyol ester', 'pentaerythritol ester',
                'fluorocarbon', 'perfluoropolyether', 'silicone oil'
            ],
            'adsorbent': [
                'zeolite', 'activated carbon', 'silica gel', 'alumina',
                'molecular sieve', 'porous polymer', 'mesoporous silica'
            ],
            'catalyst': [
                'palladium', 'platinum', 'ruthenium', 'zeolite', 'alumina'
            ]
        }
        
        return term_libraries.get(material_class, ['organic compound', 'chemical'])

    def _find_property_specialists(self, strategy: Dict, search_terms: List[str], params: Dict) -> Dict[str, List]:
        specialists = {}
        target_properties = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_properties.items():
            st.write(f"🔎 Finding {prop_name} specialists...")
            prop_terms = self._get_property_specific_terms(prop_name, search_terms)
            candidates = self._search_compounds(prop_terms, params)
            
            scored_candidates = []
            for compound in candidates:
                score = self._score_compound_for_property(compound, prop_name, criteria)
                if score > 0.3:
                    scored_candidates.append((compound, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            specialists[prop_name] = [c[0] for c in scored_candidates[:5]]
        
        return specialists

    def _find_balanced_compounds(self, strategy: Dict, search_terms: List[str], params: Dict) -> List:
        st.write("🎯 Finding balanced performers...")
        candidates = self._search_compounds(search_terms, params)
        
        scored_candidates = []
        for compound in candidates:
            overall_score = self._calculate_overall_score(compound, strategy)
            if overall_score > 0.4:
                scored_candidates.append((compound, overall_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:10]

    def _search_compounds(self, search_terms: List[str], params: Dict) -> List:
        all_compounds = []
        terms_searched = 0
        
        for term in search_terms[:params['max_terms']]:
            try:
                cache_key = f"{term}_{params['compounds_per_term']}"
                if cache_key in self.search_cache:
                    compounds = self.search_cache[cache_key]
                else:
                    compounds = pcp.get_compounds(term, 'name', listkey_count=params['compounds_per_term'])
                    self.search_cache[cache_key] = compounds
                    time.sleep(self.rate_limit_delay)
                
                valid_compounds = []
                for compound in compounds:
                    if self._is_valid_compound(compound):
                        valid_compounds.append(compound)
                
                all_compounds.extend(valid_compounds)
                terms_searched += 1
                st.write(f"   ✅ '{term}': found {len(valid_compounds)} compounds")
                
                if len(all_compounds) >= params['max_compounds']:
                    break
                    
            except Exception as e:
                st.write(f"   ❌ '{term}': search failed - {str(e)}")
                continue
        
        unique_compounds = []
        seen_cids = set()
        
        for compound in all_compounds:
            if compound.cid not in seen_cids:
                unique_compounds.append(compound)
                seen_cids.add(compound.cid)
        
        return unique_compounds[:params['max_compounds']]

    def _is_valid_compound(self, compound) -> bool:
        if not compound.molecular_weight:
            return False
        
        if compound.molecular_weight < 50 or compound.molecular_weight > 2000:
            return False
        
        formula = getattr(compound, 'molecular_formula', '')
        if not formula:
            return False
        
        if 'C' not in formula or 'H' not in formula:
            return False
        
        return True

    def _get_property_specific_terms(self, property_name: str, base_terms: List[str]) -> List[str]:
        property_enhanced_terms = {
            'thermal_conductivity': ['silicone', 'siloxane', 'ethylene glycol', 'mineral oil'],
            'viscosity': ['low viscosity', 'light oil', 'fluid'],
            'flash_point': ['high flash point', 'fire resistant', 'non flammable'],
            'surface_area': ['porous', 'activated', 'zeolite', 'MOF', 'mesoporous'],
            'adsorption_capacity': ['adsorbent', 'molecular sieve', 'activated carbon']
        }
        
        enhanced = property_enhanced_terms.get(property_name, [])
        return enhanced + base_terms[:3]

    def _score_compound_for_property(self, compound, property_name: str, criteria: Dict) -> float:
        score = 0.0
        compound_props = self._get_compound_properties(compound)
        
        if property_name == 'thermal_conductivity':
            if compound.molecular_weight:
                ideal_mw = 300
                mw_score = 1 - abs(compound.molecular_weight - ideal_mw) / 1000
                score = max(0, mw_score)
                
                if any(term in str(compound.iupac_name).lower() for term in ['siloxane', 'silicone']):
                    score += 0.3
                if 'ester' in str(compound.iupac_name).lower():
                    score += 0.2
        
        elif property_name == 'viscosity':
            if compound.molecular_weight:
                viscosity_estimate = compound.molecular_weight / 100
                target_viscosity = criteria.get('max', 50)
                if viscosity_estimate <= target_viscosity:
                    score = 1.0
                else:
                    score = target_viscosity / viscosity_estimate
                
                if compound.molecular_formula and 'C' in compound.molecular_formula:
                    carbon_count = self._count_element(compound.molecular_formula, 'C')
                    hydrogen_count = self._count_element(compound.molecular_formula, 'H')
                    if hydrogen_count / carbon_count > 1.8:
                        score += 0.2
        
        elif property_name == 'flash_point':
            if compound.molecular_weight:
                flash_point_estimate = 50 + (compound.molecular_weight * 0.3)
                min_flash_point = criteria.get('min', 150)
                if flash_point_estimate >= min_flash_point:
                    score = 1.0
                else:
                    score = flash_point_estimate / min_flash_point
        
        elif property_name == 'surface_area':
            if any(term in str(compound.iupac_name).lower() for term in ['zeolite', 'porous', 'activated', 'MOF']):
                score = 0.8
            else:
                if compound.molecular_weight:
                    complexity = min(1.0, compound.molecular_weight / 1000)
                    score = complexity * 0.5
        
        return min(1.0, score)

    def _calculate_overall_score(self, compound, strategy: Dict) -> float:
        total_score = 0
        total_weight = 0
        
        target_properties = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_properties.items():
            weight = criteria.get('weight', 0.1)
            prop_score = self._score_compound_for_property(compound, prop_name, criteria)
            
            total_score += prop_score * weight
            total_weight += weight
        
        constraints = strategy.get('safety_constraints', [])
        constraint_penalty = self._check_constraints(compound, constraints)
        
        overall_score = total_score / max(total_weight, 0.1)
        return max(0, overall_score - constraint_penalty)

    def _check_constraints(self, compound, constraints: List[str]) -> float:
        penalty = 0.0
        formula = getattr(compound, 'molecular_formula', '')
        
        for constraint in constraints:
            if constraint == 'non_toxic':
                if any(element in formula for element in ['Pb', 'Hg', 'Cd', 'As']):
                    penalty += 0.5
                if 'Cl' in formula and formula.count('Cl') > 2:
                    penalty += 0.3
            
            elif constraint == 'pfas_free':
                if 'F' in formula:
                    penalty += 0.4
            
            elif constraint == 'biodegradable':
                if compound.molecular_weight and compound.molecular_weight > 500:
                    penalty += 0.2
                if any(term in str(compound.iupac_name).lower() for term in ['siloxane', 'silicone']):
                    penalty += 0.3
        
        return penalty

    def _get_compound_properties(self, compound) -> Dict:
        if compound.cid in self.property_cache:
            return self.property_cache[compound.cid]
        
        properties = {
            'molecular_weight': compound.molecular_weight,
            'formula': compound.molecular_formula,
            'iupac_name': compound.iupac_name,
            'smiles': getattr(compound, 'canonical_smiles', '')
        }
        
        self.property_cache[compound.cid] = properties
        return properties

    def _count_element(self, formula: str, element: str) -> int:
        import re
        pattern = f"{element}(\\d+)"
        matches = re.findall(pattern, formula)
        return sum(int(match) for match in matches) if matches else (1 if element in formula else 0)
