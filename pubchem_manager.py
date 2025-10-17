# pubchem_manager.py - Advanced PubChem integration
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
        self.rate_limit_delay = 0.3  # seconds between API calls
        
    def find_compounds(self, strategy: Dict, max_compounds: int, search_depth: str) -> Dict[str, Any]:
        """Find compounds using intelligent search strategy"""
        
        # Adjust search parameters based on depth
        search_params = self._get_search_parameters(search_depth, max_compounds)
        
        # Get search terms from strategy
        search_terms = strategy.get('search_strategy', {}).get('search_terms', [])
        chemical_classes = strategy.get('search_strategy', {}).get('chemical_classes', [])
        
        if not search_terms:
            search_terms = self._generate_default_search_terms(strategy['material_class'])
        
        st.info(f"🔍 Searching for: {', '.join(search_terms[:5])}...")
        
        # Find specialists for each target property
        specialists = self._find_property_specialists(strategy, search_terms, search_params)
        
        # Find balanced compounds
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
        """Get search parameters based on depth"""
        params = {
            "Quick Scan": {
                'compounds_per_term': 5,
                'max_terms': 4,
                'timeout': 30
            },
            "Standard Analysis": {
                'compounds_per_term': 10, 
                'max_terms': 6,
                'timeout': 60
            },
            "Comprehensive Search": {
                'compounds_per_term': 15,
                'max_terms': 8,
                'timeout': 120
            }
        }
        
        base_params = params.get(search_depth, params["Standard Analysis"])
        base_params['max_compounds'] = max_compounds
        return base_params
    
    def _generate_default_search_terms(self, material_class: str) -> List[str]:
        """Generate default search terms based on material class"""
        term_libraries = {
            'coolant': [
                'siloxane', 'polydimethylsiloxane', 'polyalphaolefin', 'PAO',
                'mineral oil', 'synthetic ester', 'polyol ester', 'propylene glycol',
                'ethylene glycol', 'paraffin oil', 'alkanes', 'silicone oil'
            ],
            'adsorbent': [
                'zeolite', 'activated carbon', 'silica gel', 'MOF', 'metal organic framework',
                'porous polymer', 'mesoporous silica', 'alumina', 'clay', 'carbon molecular sieve'
            ],
            'catalyst': [
                'palladium', 'platinum', 'ruthenium', 'rhodium', 'zeolite', 'alumina',
                'silica', 'titania', 'zirconia', 'molecular sieve', 'heterogeneous catalyst'
            ],
            'polymer': [
                'polyethylene', 'polypropylene', 'polystyrene', 'PVC', 'nylon',
                'polycarbonate', 'PET', 'PTFE', 'polyurethane', 'epoxy resin'
            ]
        }
        
        return term_libraries.get(material_class, ['organic compound', 'industrial chemical'])
    
    def _find_property_specialists(self, strategy: Dict, search_terms: List[str], params: Dict) -> Dict[str, List]:
        """Find compounds that excel in specific properties"""
        specialists = {}
        target_properties = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_properties.items():
            st.write(f"🔎 Finding {prop_name} specialists...")
            
            # Get property-specific search terms
            prop_terms = self._get_property_specific_terms(prop_name, search_terms)
            
            # Search for compounds
            candidates = self._search_compounds(prop_terms, params)
            
            # Score compounds for this specific property
            scored_candidates = []
            for compound in candidates:
                score = self._score_compound_for_property(compound, prop_name, criteria)
                if score > 0.3:  # Minimum threshold
                    scored_candidates.append((compound, score))
            
            # Sort by property-specific score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            specialists[prop_name] = [c[0] for c in scored_candidates[:5]]  # Top 5 specialists
        
        return specialists
    
    def _find_balanced_compounds(self, strategy: Dict, search_terms: List[str], params: Dict) -> List:
        """Find compounds with good overall balanced performance"""
        st.write("🎯 Finding balanced performers...")
        
        # Search for compounds
        candidates = self._search_compounds(search_terms, params)
        
        # Score compounds for overall performance
        scored_candidates = []
        for compound in candidates:
            overall_score = self._calculate_overall_score(compound, strategy)
            if overall_score > 0.4:  # Minimum threshold
                scored_candidates.append((compound, overall_score))
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:10]  # Top 10 balanced candidates
    
    def _search_compounds(self, search_terms: List[str], params: Dict) -> List:
        """Perform actual PubChem searches"""
        all_compounds = []
        terms_searched = 0
        
        for term in search_terms[:params['max_terms']]:
            try:
                # Check cache first
                cache_key = f"{term}_{params['compounds_per_term']}"
                if cache_key in self.search_cache:
                    compounds = self.search_cache[cache_key]
                else:
                    # Perform PubChem search
                    compounds = pcp.get_compounds(term, 'name', listkey_count=params['compounds_per_term'])
                    self.search_cache[cache_key] = compounds
                    time.sleep(self.rate_limit_delay)  # Rate limiting
                
                # Filter valid compounds
                valid_compounds = []
                for compound in compounds:
                    if self._is_valid_compound(compound):
                        valid_compounds.append(compound)
                
                all_compounds.extend(valid_compounds)
                terms_searched += 1
                
                st.write(f"   ✅ '{term}': found {len(valid_compounds)} compounds")
                
                # Stop if we have enough compounds
                if len(all_compounds) >= params['max_compounds']:
                    break
                    
            except Exception as e:
                st.write(f"   ❌ '{term}': search failed - {str(e)}")
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
        """Check if compound is valid for materials applications"""
        # Basic validity checks
        if not compound.molecular_weight:
            return False
        
        # Filter out very small or very large molecules
        if compound.molecular_weight < 50 or compound.molecular_weight > 2000:
            return False
        
        # Filter out inorganic and organometallic unless specifically needed
        formula = getattr(compound, 'molecular_formula', '')
        if not formula:
            return False
        
        # Basic organic compound check (contains C and H)
        if 'C' not in formula or 'H' not in formula:
            return False
        
        return True
    
    def _get_property_specific_terms(self, property_name: str, base_terms: List[str]) -> List[str]:
        """Get search terms optimized for specific properties"""
        property_enhanced_terms = {
            'thermal_conductivity': ['silicone', 'siloxane', 'ethylene glycol', 'mineral oil'],
            'viscosity': ['low viscosity', 'light oil', 'fluid'],
            'flash_point': ['high flash point', 'fire resistant', 'non flammable'],
            'surface_area': ['porous', 'activated', 'zeolite', 'MOF', 'mesoporous'],
            'adsorption_capacity': ['adsorbent', 'molecular sieve', 'activated carbon']
        }
        
        enhanced = property_enhanced_terms.get(property_name, [])
        return enhanced + base_terms[:3]  # Combine with base terms
    
    def _score_compound_for_property(self, compound, property_name: str, criteria: Dict) -> float:
        """Score compound for a specific property"""
        score = 0.0
        
        # Get compound properties
        compound_props = self._get_compound_properties(compound)
        
        if property_name == 'thermal_conductivity':
            # Estimate based on molecular structure
            if compound.molecular_weight:
                # Simple heuristic: medium MW compounds often have good thermal conductivity
                ideal_mw = 300
                mw_score = 1 - abs(compound.molecular_weight - ideal_mw) / 1000
                score = max(0, mw_score)
                
                # Silicones and certain esters tend to have better thermal conductivity
                if any(term in str(compound.iupac_name).lower() for term in ['siloxane', 'silicone']):
                    score += 0.3
                if 'ester' in str(compound.iupac_name).lower():
                    score += 0.2
        
        elif property_name == 'viscosity':
            # Estimate viscosity based on molecular weight and structure
            if compound.molecular_weight:
                # Lower MW generally means lower viscosity
                viscosity_estimate = compound.molecular_weight / 100
                target_viscosity = criteria.get('max', 50)
                if viscosity_estimate <= target_viscosity:
                    score = 1.0
                else:
                    score = target_viscosity / viscosity_estimate
                
                # Linear molecules generally have lower viscosity
                if compound.molecular_formula and 'C' in compound.molecular_formula:
                    carbon_count = self._count_element(compound.molecular_formula, 'C')
                    hydrogen_count = self._count_element(compound.molecular_formula, 'H')
                    if hydrogen_count / carbon_count > 1.8:  # More saturated, likely lower viscosity
                        score += 0.2
        
        elif property_name == 'flash_point':
            # Estimate flash point based on molecular weight
            if compound.molecular_weight:
                flash_point_estimate = 50 + (compound.molecular_weight * 0.3)
                min_flash_point = criteria.get('min', 150)
                if flash_point_estimate >= min_flash_point:
                    score = 1.0
                else:
                    score = flash_point_estimate / min_flash_point
        
        elif property_name == 'surface_area':
            # Porous materials have high surface area
            if any(term in str(compound.iupac_name).lower() for term in ['zeolite', 'porous', 'activated', 'MOF']):
                score = 0.8
            else:
                # Estimate based on molecular structure complexity
                if compound.molecular_weight:
                    complexity = min(1.0, compound.molecular_weight / 1000)
                    score = complexity * 0.5
        
        return min(1.0, score)
    
    def _calculate_overall_score(self, compound, strategy: Dict) -> float:
        """Calculate overall score considering all target properties"""
        total_score = 0
        total_weight = 0
        
        target_properties = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_properties.items():
            weight = criteria.get('weight', 0.1)
            prop_score = self._score_compound_for_property(compound, prop_name, criteria)
            
            total_score += prop_score * weight
            total_weight += weight
        
        # Apply constraints penalty
        constraints = strategy.get('safety_constraints', [])
        constraint_penalty = self._check_constraints(compound, constraints)
        
        overall_score = total_score / max(total_weight, 0.1)
        return max(0, overall_score - constraint_penalty)
    
    def _check_constraints(self, compound, constraints: List[str]) -> float:
        """Check safety and other constraints"""
        penalty = 0.0
        formula = getattr(compound, 'molecular_formula', '')
        
        for constraint in constraints:
            if constraint == 'non_toxic':
                # Simple heuristic: avoid heavy metals and known toxic groups
                if any(element in formula for element in ['Pb', 'Hg', 'Cd', 'As']):
                    penalty += 0.5
                if 'Cl' in formula and formula.count('Cl') > 2:  # Highly chlorinated
                    penalty += 0.3
            
            elif constraint == 'pfas_free':
                if 'F' in formula:  # Simple check for fluorine
                    penalty += 0.4
            
            elif constraint == 'biodegradable':
                # Complex molecules less likely to be biodegradable
                if compound.molecular_weight and compound.molecular_weight > 500:
                    penalty += 0.2
                if any(term in str(compound.iupac_name).lower() for term in ['siloxane', 'silicone']):
                    penalty += 0.3  # Silicones less biodegradable
        
        return penalty
    
    def _get_compound_properties(self, compound) -> Dict:
        """Get compound properties with caching"""
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
        """Count occurrences of an element in molecular formula"""
        import re
        pattern = f"{element}(\\d+)"
        matches = re.findall(pattern, formula)
        return sum(int(match) for match in matches) if matches else (1 if element in formula else 0)
