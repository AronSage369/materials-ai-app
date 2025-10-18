# pubchem_manager.py - REWRITTEN FOR EFFICIENCY AND RELIABILITY
import pubchempy as pcp
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any
import streamlit as st

class PubChemManager:
    def __init__(self):
        """
        Initializes the PubChemManager with a small, reliable internal database.
        This database acts as a fallback if live PubChem searches fail.
        """
        self.known_compounds_db = {
            'Coolant/Lubricant': [
                {'cid': 24764, 'iupac_name': 'Hexamethyldisiloxane', 'molecular_formula': 'C6H18OSi2', 'molecular_weight': '162.4'},
                {'cid': 24705, 'iupac_name': 'Octamethyltrisiloxane', 'molecular_formula': 'C8H24O2Si3', 'molecular_weight': '236.5'},
                {'cid': 1030, 'iupac_name': 'Propylene Glycol', 'molecular_formula': 'C3H8O2', 'molecular_weight': '76.1'},
                {'cid': 174, 'iupac_name': 'Ethylene Glycol', 'molecular_formula': 'C2H6O2', 'molecular_weight': '62.1'}
            ],
            'Adsorbent': [
                {'cid': 24844, 'iupac_name': 'Zeolite Y', 'molecular_formula': 'Na56Al56Si136O384', 'molecular_weight': '13000'},
                {'cid': 5462310, 'iupac_name': 'Activated Carbon', 'molecular_formula': 'C', 'molecular_weight': '12.0'},
                {'cid': 10978189, 'iupac_name': 'MIL-53(Al)', 'molecular_formula': 'C8H5AlO5', 'molecular_weight': '208.1'}
            ]
        }

    def find_compounds(self, strategy: Dict, max_compounds: int, search_depth: str) -> Dict[str, Any]:
        """
        Efficiently finds and scores compounds in a single pass.
        1. Generates effective search terms by blending AI and expert knowledge.
        2. Performs ONE bulk search on PubChem to create a candidate pool.
        3. Falls back to a reliable internal database if the live search fails.
        4. Scores all found compounds in-memory to identify specialists and balanced performers.
        """
        search_params = self._get_search_parameters(search_depth, max_compounds)
        material_class = strategy.get('material_class', 'coolant')
        
        # Step 1: Generate a high-quality, reliable list of search terms
        search_terms = self._generate_effective_search_terms(material_class, strategy)
        st.info(f"🔍 Searching PubChem with robust terms: {', '.join(search_terms[:4])}...")
        
        # Step 2: Perform a single, efficient bulk search for all candidate compounds
        all_candidates = self._search_compounds(search_terms, search_params)
        
        # Step 3: If the live search yields no results, use the internal fallback database
        if not all_candidates:
            st.warning("⚠️ No compounds found in primary PubChem search. Using internal database as fallback...")
            all_candidates = self._use_known_compounds_fallback(material_class)
            if not all_candidates:
                st.error("❌ Critical Error: No compounds found in PubChem or the internal fallback database.")
                return {'specialists': {}, 'balanced': [], 'search_metrics': {}}

        st.success(f"✅ Found {len(all_candidates)} potential candidate compounds for analysis.")
        
        # Step 4: Score and categorize the entire pool of candidates in-memory
        st.write("🔬 Scoring and categorizing all candidates...")
        specialists = self._find_property_specialists(strategy, all_candidates)
        balanced_candidates = self._find_balanced_compounds(strategy, all_candidates)
        
        total_found = sum(len(s) for s in specialists.values()) + len(balanced_candidates)
        
        return {
            'specialists': specialists,
            'balanced': balanced_candidates,
            'search_metrics': {
                'terms_searched': len(search_terms),
                'total_compounds_found': total_found
            }
        }

    def _generate_effective_search_terms(self, material_class: str, strategy: Dict) -> List[str]:
        """Generates a list of reliable search terms by combining AI suggestions with a predefined expert list."""
        base_terms = strategy.get('search_strategy', {}).get('search_terms', [])
        
        # Add predefined reliable terms for the material class to improve search success
        expert_terms = {
            'coolant': ["siloxane", "polyalphaolefin", "synthetic ester", "propylene glycol", "mineral oil"],
            'adsorbent': ["zeolite", "activated carbon", "metal-organic framework", "silica gel", "porous polymer"]
        }
        base_terms.extend(expert_terms.get(material_class, []))
        
        # Return a unique list of terms
        return list(dict.fromkeys(base_terms))

    def _use_known_compounds_fallback(self, material_class: str) -> List[Any]:
        """
        Uses a small internal DB as a fallback, creating mock PubChemPy objects
        to ensure compatibility with the rest of the analysis pipeline.
        """
        material_type_map = {
            'coolant': 'Coolant/Lubricant',
            'adsorbent': 'Adsorbent'
        }
        key = material_type_map.get(material_class, 'Coolant/Lubricant')
        compounds_data = self.known_compounds_db.get(key, [])
        
        # Create mock objects that mimic the structure of real pubchempy.Compound objects
        mock_compounds = [type('Compound', (object,), data)() for data in compounds_data]
        return mock_compounds

    def _get_search_parameters(self, search_depth: str, max_compounds: int) -> Dict:
        """Configures search parameters based on user-selected depth."""
        params = {"Quick Scan": 5, "Standard Analysis": 10, "Comprehensive Search": 20}
        return {'max_results_per_term': params.get(search_depth, 10), 'max_compounds': max_compounds}

    def _find_property_specialists(self, strategy: Dict, candidates: List) -> Dict[str, List]:
        """Efficiently finds specialist compounds from a pre-fetched list of candidates."""
        specialists = {}
        target_properties = strategy.get('target_properties', {})
        
        if not isinstance(target_properties, dict):
            st.warning("⚠️ Target properties from AI were not a dictionary. Cannot find specialists.")
            return {}
        
        for prop_name, criteria in target_properties.items():
            scored_candidates = []
            for compound in candidates:
                score = self._score_compound_for_property(compound, prop_name, criteria)
                if score > 0.6:  # Set a higher threshold for a compound to be considered a "specialist"
                    scored_candidates.append((compound, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            specialists[prop_name] = [c[0] for c in scored_candidates[:5]]  # Return the top 5 specialists
        
        return specialists

    def _find_balanced_compounds(self, strategy: Dict, candidates: List) -> List:
        """Efficiently finds balanced performers from a pre-fetched list of candidates."""
        scored_candidates = []
        for compound in candidates:
            overall_score = self._calculate_overall_score(compound, strategy)
            if overall_score > 0.5: # Threshold to be considered a viable balanced performer
                scored_candidates.append((compound, overall_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:50] # Return up to 50 top balanced candidates for formulation

    def _search_compounds(self, search_terms: List[str], params: Dict) -> List:
        """Performs a single, efficient bulk search across all terms."""
        all_compounds = []
        cids_seen = set()
        
        for term in search_terms:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=params['max_results_per_term'])
                for compound in results:
                    if self._is_valid_compound(compound) and compound.cid not in cids_seen:
                        all_compounds.append(compound)
                        cids_seen.add(compound.cid)
                time.sleep(0.1) # Be respectful to the PubChem API
                if len(all_compounds) >= params['max_compounds']:
                    break
            except Exception:
                continue # Gracefully ignore errors for individual search terms
        
        return all_compounds[:params['max_compounds']]

    def _is_valid_compound(self, compound) -> bool:
        """Performs basic validation on a compound object to ensure it has the necessary data."""
        return (hasattr(compound, 'cid') and compound.cid is not None and
                hasattr(compound, 'molecular_weight') and compound.molecular_weight is not None)

    def _score_compound_for_property(self, compound, property_name: str, criteria: Dict) -> float:
        """
        Generic scoring based on heuristics if no real data is available.
        This is a placeholder for a more advanced machine learning model.
        """
        try:
            mw = float(compound.molecular_weight)
            if mw > 1000: return 0.3 # Penalize very large molecules
            
            # Simple heuristic scoring based on property name and molecular weight
            if any(p in property_name for p in ['conductivity', 'heat']):
                return max(0.2, 1 - (abs(mw - 250) / 500)) # Optimal around 250 g/mol
            if 'viscosity' in property_name:
                return max(0.2, (mw / 400))
            if 'flash_point' in property_name:
                 return max(0.2, (mw / 500))
            if 'surface_area' in property_name: # For adsorbents
                 return max(0.2, 1 - (abs(mw - 400) / 800))
            
            return 0.5 # Default score for unknown properties
        except (ValueError, TypeError):
            return 0.2

    def _calculate_overall_score(self, compound, strategy: Dict) -> float:
        """Calculates a weighted overall score for a compound based on target properties."""
        total_score, total_weight = 0, 0
        target_properties = strategy.get('target_properties', {})
        
        if not isinstance(target_properties, dict): return 0.5
        
        for prop_name, criteria in target_properties.items():
            weight = criteria.get('weight', 0.1)
            prop_score = self._score_compound_for_property(compound, prop_name, criteria)
            total_score += prop_score * weight
            total_weight += weight
        
        # Apply a penalty for any safety constraint violations
        penalty = self._check_constraints(compound, strategy.get('safety_constraints', []))
        
        overall_score = (total_score / max(total_weight, 0.1)) - penalty
        return max(0.1, overall_score) # Ensure a minimum score

    def _check_constraints(self, compound, constraints: List[str]) -> float:
        """Applies a penalty for safety constraint violations based on compound name."""
        penalty = 0.0
        name = str(getattr(compound, 'iupac_name', '')).lower()
        if 'pfas_free' in constraints and 'fluoro' in name:
            penalty += 0.3
        if 'non_toxic' in constraints and any(term in name for term in ['toxic', 'poison', 'hazard']):
            penalty += 0.3
        return penalty

