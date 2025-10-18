# pubchem_manager.py - IMPROVED WITH FALLBACKS
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
        
        # Known coolant compounds as fallback (PubChem CIDs)
        self.known_coolants = {
            'silicones': [
                6390,   # Hexamethyldisiloxane
                11403,  # Octamethyltrisiloxane  
                99649,  # Decamethylcyclopentasiloxane
                24755,  # Polydimethylsiloxane
            ],
            'esters': [
                31261,  # Dioctyl sebacate
                8186,   # Dibutyl sebacate
                31255,  # Diethyl sebacate
            ],
            'oils': [
                32555,  # Mineral oil
                8036,   # Paraffin oil
                8180,   # White oil
            ],
            'glycols': [
                174,    # Ethylene glycol
                1030,   # Propylene glycol
                12139,  # Polyethylene glycol
            ],
            'hydrocarbons': [
                12345,  # Polyalphaolefin (example)
                8123,   # Alkylbenzene
            ]
        }

    def find_compounds(self, strategy: Dict, max_compounds: int, search_depth: str) -> Dict[str, Any]:
        search_params = self._get_search_parameters(search_depth, max_compounds)
        
        search_terms = strategy.get('search_strategy', {}).get('search_terms', [])
        
        if not search_terms:
            search_terms = self._generate_effective_search_terms(strategy['material_class'])
        
        st.info(f"🔍 Searching with terms: {', '.join(search_terms[:5])}...")
        
        # Try primary search
        specialists = self._find_property_specialists(strategy, search_terms, search_params)
        balanced_candidates = self._find_balanced_compounds(strategy, search_terms, search_params)
        
        # If no results, use fallback known compounds
        total_found = sum(len(s) for s in specialists.values()) + len(balanced_candidates)
        if total_found == 0:
            st.warning("⚠️ No compounds found in search. Using known coolant database...")
            return self._use_known_compounds_fallback(strategy, max_compounds)
        
        return {
            'specialists': specialists,
            'balanced': balanced_candidates,
            'search_metrics': {
                'terms_searched': len(search_terms),
                'total_compounds_found': total_found
            }
        }

    def _generate_effective_search_terms(self, material_class: str) -> List[str]:
        """Generate search terms that actually work in PubChem"""
        if material_class == 'coolant':
            return [
                # Specific compounds that exist in PubChem
                'hexamethyldisiloxane', 'octamethyltrisiloxane', 'decamethylcyclopentasiloxane',
                'polydimethylsiloxane', 'dimethyl siloxane',
                'mineral oil', 'paraffin oil', 'white oil',
                'ethylene glycol', 'propylene glycol', 'polyethylene glycol',
                'dioctyl sebacate', 'dibutyl sebacate', 'diethyl sebacate',
                'polyalphaolefin', 'alkylbenzene', 'synthetic oil',
                'ester oil', 'polyol ester', 'pentaerythritol ester',
                'silicone oil', 'fluorocarbon', 'perfluoropolyether'
            ]
        else:
            return ['organic compound', 'polymer', 'industrial chemical']

    def _use_known_compounds_fallback(self, strategy: Dict, max_compounds: int) -> Dict[str, Any]:
        """Use known compounds when searches fail"""
        specialists = {}
        balanced_candidates = []
        
        # Get all known coolant CIDs
        all_cids = []
        for category, cids in self.known_coolants.items():
            all_cids.extend(cids)
        
        # Fetch compounds and score them
        scored_compounds = []
        for cid in all_cids[:max_compounds]:
            try:
                compound = pcp.Compound.from_cid(cid)
                if self._is_valid_compound(compound):
                    score = self._calculate_overall_score(compound, strategy)
                    scored_compounds.append((compound, score))
                    time.sleep(0.2)  # Rate limiting
            except Exception as e:
                st.write(f"   ⚠️ Could not fetch CID {cid}: {e}")
                continue
        
        # Sort by score
        scored_compounds.sort(key=lambda x: x[1], reverse=True)
        balanced_candidates = scored_compounds
        
        # Create specialists from top compounds
        target_properties = strategy.get('target_properties', {})
        for prop_name in target_properties.keys():
            # Just use top balanced candidates as specialists for now
            specialists[prop_name] = [c[0] for c in scored_compounds[:3]]
        
        st.success(f"✅ Found {len(scored_compounds)} known coolant compounds")
        
        return {
            'specialists': specialists,
            'balanced': balanced_candidates,
            'search_metrics': {
                'terms_searched': 0,
                'total_compounds_found': len(scored_compounds),
                'fallback_used': True
            }
        }

    def _get_search_parameters(self, search_depth: str, max_compounds: int) -> Dict:
        params = {
            "Quick Scan": {'compounds_per_term': 8, 'max_terms': 6, 'timeout': 30},
            "Standard Analysis": {'compounds_per_term': 12, 'max_terms': 8, 'timeout': 60},
            "Comprehensive Search": {'compounds_per_term': 15, 'max_terms': 10, 'timeout': 120}
        }
        
        base_params = params.get(search_depth, params["Standard Analysis"])
        base_params['max_compounds'] = max_compounds
        return base_params

    def _find_property_specialists(self, strategy: Dict, search_terms: List[str], params: Dict) -> Dict[str, List]:
        specialists = {}
        target_properties = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_properties.items():
            st.write(f"🔎 Finding {prop_name} specialists...")
            
            # Use all search terms, not just property-specific ones
            candidates = self._search_compounds(search_terms, params)
            
            scored_candidates = []
            for compound in candidates:
                score = self._score_compound_for_property(compound, prop_name, criteria)
                if score > 0.2:  # Lower threshold
                    scored_candidates.append((compound, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            specialists[prop_name] = [c[0] for c in scored_candidates[:3]]  # Top 3 specialists
        
        return specialists

    def _find_balanced_compounds(self, strategy: Dict, search_terms: List[str], params: Dict) -> List:
        st.write("🎯 Finding balanced performers...")
        candidates = self._search_compounds(search_terms, params)
        
        scored_candidates = []
        for compound in candidates:
            overall_score = self._calculate_overall_score(compound, strategy)
            if overall_score > 0.3:  # Lower threshold
                scored_candidates.append((compound, overall_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:15]  # Return more candidates

    def _search_compounds(self, search_terms: List[str], params: Dict) -> List:
        all_compounds = []
        terms_searched = 0
        
        for term in search_terms[:params['max_terms']]:
            try:
                cache_key = f"{term}_{params['compounds_per_term']}"
                if cache_key in self.search_cache:
                    compounds = self.search_cache[cache_key]
                else:
                    # Try different search types
                    compounds = []
                    
                    # First try name search
                    try:
                        compounds = pcp.get_compounds(term, 'name', listkey_count=params['compounds_per_term'])
                    except:
                        pass
                    
                    # If no results, try synonym search
                    if not compounds:
                        try:
                            compounds = pcp.get_compounds(term, 'synonym', listkey_count=params['compounds_per_term'])
                        except:
                            pass
                    
                    self.search_cache[cache_key] = compounds
                    time.sleep(self.rate_limit_delay)
                
                valid_compounds = []
                for compound in compounds:
                    if self._is_valid_compound(compound):
                        valid_compounds.append(compound)
                
                all_compounds.extend(valid_compounds)
                terms_searched += 1
                
                if valid_compounds:
                    st.write(f"   ✅ '{term}': found {len(valid_compounds)} compounds")
                else:
                    st.write(f"   ❌ '{term}': no compounds found")
                
                if len(all_compounds) >= params['max_compounds']:
                    break
                    
            except Exception as e:
                st.write(f"   ⚠️ '{term}': search error")
                continue
        
        # Remove duplicates by CID
        unique_compounds = []
        seen_cids = set()
        
        for compound in all_compounds:
            if compound.cid not in seen_cids:
                unique_compounds.append(compound)
                seen_cids.add(compound.cid)
        
        return unique_compounds[:params['max_compounds']]

    def _is_valid_compound(self, compound) -> bool:
        """More lenient validation for coolants"""
        if not compound.molecular_weight:
            return False
        
        # Wider range for coolants
        if compound.molecular_weight < 40 or compound.molecular_weight > 3000:
            return False
        
        formula = getattr(compound, 'molecular_formula', '')
        if not formula:
            return False
        
        # Allow some inorganic coolants
        if 'C' not in formula and 'Si' not in formula:
            return False
        
        return True

    def _score_compound_for_property(self, compound, property_name: str, criteria: Dict) -> float:
        score = 0.0
        
        if property_name == 'thermal_conductivity':
            if compound.molecular_weight:
                # Silicones and esters generally have better thermal conductivity
                name_lower = str(compound.iupac_name).lower() if compound.iupac_name else ""
                
                if any(term in name_lower for term in ['siloxane', 'silicone']):
                    score = 0.8
                elif any(term in name_lower for term in ['ester', 'glycol']):
                    score = 0.7
                elif any(term in name_lower for term in ['oil', 'alkane']):
                    score = 0.6
                else:
                    score = 0.5
        
        elif property_name == 'viscosity':
            if compound.molecular_weight:
                # Lower MW generally means lower viscosity
                if compound.molecular_weight < 300:
                    score = 0.9
                elif compound.molecular_weight < 500:
                    score = 0.7
                elif compound.molecular_weight < 800:
                    score = 0.5
                else:
                    score = 0.3
        
        elif property_name == 'flash_point':
            if compound.molecular_weight:
                # Higher MW generally means higher flash point
                if compound.molecular_weight > 400:
                    score = 0.9
                elif compound.molecular_weight > 250:
                    score = 0.7
                elif compound.molecular_weight > 150:
                    score = 0.5
                else:
                    score = 0.3
        
        elif property_name == 'specific_heat':
            # Oxygenated compounds often have higher specific heat
            formula = getattr(compound, 'molecular_formula', '')
            name_lower = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if 'O' in formula and formula.count('O') >= 2:
                score = 0.8
            elif any(term in name_lower for term in ['glycol', 'alcohol']):
                score = 0.9
            elif any(term in name_lower for term in ['ester', 'ether']):
                score = 0.7
            else:
                score = 0.5
        
        elif property_name == 'dielectric_strength':
            # Non-polar compounds generally have better dielectric strength
            formula = getattr(compound, 'molecular_formula', '')
            name_lower = str(compound.iupac_name).lower() if compound.iupac_name else ""
            
            if 'O' not in formula and 'N' not in formula:  # Hydrocarbons
                score = 0.9
            elif any(term in name_lower for term in ['siloxane', 'silicone']):
                score = 0.8
            else:
                score = 0.6
        
        else:
            # Default scoring for other properties
            score = 0.5
        
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
        
        # Apply safety constraints
        constraints = strategy.get('safety_constraints', [])
        constraint_penalty = self._check_constraints(compound, constraints)
        
        overall_score = total_score / max(total_weight, 0.1)
        return max(0.1, overall_score - constraint_penalty)  # Minimum 0.1 score

    def _check_constraints(self, compound, constraints: List[str]) -> float:
        penalty = 0.0
        formula = getattr(compound, 'molecular_formula', '')
        name_lower = str(compound.iupac_name).lower() if compound.iupac_name else ""
        
        for constraint in constraints:
            if constraint == 'non_toxic':
                # Simple heuristic - avoid heavy metals and highly chlorinated
                if any(element in formula for element in ['Pb', 'Hg', 'Cd', 'As']):
                    penalty += 0.8
                elif 'Cl' in formula and formula.count('Cl') > 3:
                    penalty += 0.4
            
            elif constraint == 'pfas_free':
                if 'F' in formula and any(term in name_lower for term in ['fluoro', 'perfluoro']):
                    penalty += 0.6
                elif 'F' in formula:
                    penalty += 0.3  # Minor penalty for any fluorine
            
            elif constraint == 'biodegradable':
                # Complex molecules and silicones less biodegradable
                if compound.molecular_weight and compound.molecular_weight > 600:
                    penalty += 0.3
                if any(term in name_lower for term in ['siloxane', 'silicone']):
                    penalty += 0.4
        
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
