import pubchempy as pcp
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import random
import time
import logging
import numpy as np
from utils import cached

class IntelligentPubChemSearcher:
    """
    AI-driven PubChem search that uses intelligent strategies to find innovative compounds
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @cached
    def ai_driven_compound_search(self, strategy: Dict, material_type: str, max_compounds: int = 200) -> List[Dict]:
        """
        Use AI-generated strategy to perform intelligent compound search
        """
        self.logger.info(f"🚀 Starting AI-driven search for {material_type}")
        
        all_compounds = []
        search_strategy = strategy.get('search_strategy', {})
        
        # Search across all AI-suggested categories
        search_categories = [
            *search_strategy.get('electronic_properties', []),
            *search_strategy.get('structural_features', []),
            *search_strategy.get('functional_groups', []),
            *search_strategy.get('application_terms', []),
            *search_strategy.get('innovative_concepts', [])
        ]
        
        # Remove duplicates and limit
        search_terms = list(set(search_categories))[:30]
        
        self.logger.info(f"🔍 Search terms: {search_terms}")
        
        # Multi-strategy search
        for strategy_name, search_method in [
            ('electronic', self._search_electronic_materials),
            ('structural', self._search_structural_features),
            ('functional', self._search_functional_groups),
            ('innovative', self._search_innovative_concepts)
        ]:
            try:
                compounds = search_method(strategy, search_terms, max_compounds // 4)
                all_compounds.extend(compounds)
                self.logger.info(f"✅ {strategy_name} search found {len(compounds)} compounds")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                self.logger.error(f"❌ {strategy_name} search failed: {e}")
        
        # Remove duplicates and filter
        unique_compounds = self._remove_duplicates(all_compounds)
        filtered_compounds = [c for c in unique_compounds if self._advanced_filter(c, strategy)]
        
        # Enhanced scoring
        scored_compounds = self._ai_enhanced_scoring(filtered_compounds, strategy)
        
        self.logger.info(f"🎯 Total unique compounds: {len(scored_compounds)}")
        return scored_compounds[:max_compounds]

    def _search_electronic_materials(self, strategy: Dict, search_terms: List[str], max_results: int) -> List[Dict]:
        """Search for electronic and optical materials"""
        compounds = []
        electronic_terms = strategy.get('search_strategy', {}).get('electronic_properties', [])
        
        for term in electronic_terms[:10]:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=15)
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data and self._has_electronic_potential(comp_data):
                        compounds.append(comp_data)
                time.sleep(0.3)
            except Exception as e:
                self.logger.warning(f"Electronic search failed for {term}: {e}")
        
        return compounds

    def _search_structural_features(self, strategy: Dict, search_terms: List[str], max_results: int) -> List[Dict]:
        """Search based on structural features"""
        compounds = []
        structural_terms = strategy.get('search_strategy', {}).get('structural_features', [])
        
        for term in structural_terms[:10]:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=12)
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        compounds.append(comp_data)
                time.sleep(0.3)
            except Exception as e:
                self.logger.warning(f"Structural search failed for {term}: {e}")
        
        return compounds

    def _search_functional_groups(self, strategy: Dict, search_terms: List[str], max_results: int) -> List[Dict]:
        """Search based on functional groups"""
        compounds = []
        functional_terms = strategy.get('search_strategy', {}).get('functional_groups', [])
        
        for term in functional_terms[:10]:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=12)
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data and self._has_relevant_functionality(comp_data, strategy):
                        compounds.append(comp_data)
                time.sleep(0.3)
            except Exception as e:
                self.logger.warning(f"Functional search failed for {term}: {e}")
        
        return compounds

    def _search_innovative_concepts(self, strategy: Dict, search_terms: List[str], max_results: int) -> List[Dict]:
        """Search for innovative and emerging materials"""
        compounds = []
        innovative_terms = strategy.get('search_strategy', {}).get('innovative_concepts', [])
        
        for term in innovative_terms[:10]:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=10)
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data and self._is_innovative(comp_data):
                        compounds.append(comp_data)
                time.sleep(0.3)
            except Exception as e:
                self.logger.warning(f"Innovative search failed for {term}: {e}")
        
        return compounds

    def _extract_compound_data(self, compound) -> Optional[Dict]:
        """Extract comprehensive compound data"""
        try:
            properties = ['MolecularWeight', 'MolecularFormula', 'CanonicalSMILES', 
                         'IUPACName', 'XLogP', 'Complexity', 'HydrogenBondDonorCount',
                         'HydrogenBondAcceptorCount', 'RotatableBondCount', 'TPSA']
            
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
                'hbd_count': comp_props.get('HydrogenBondDonorCount'),
                'hba_count': comp_props.get('HydrogenBondAcceptorCount'),
                'rotatable_bonds': comp_props.get('RotatableBondCount'),
                'tpsa': comp_props.get('TPSA'),
                'synonyms': compound.synonyms[:5] if compound.synonyms else [],
                'source': 'pubchem'
            }
        except Exception as e:
            self.logger.error(f"Error extracting compound data: {e}")
            return None

    def _advanced_filter(self, compound: Dict, strategy: Dict) -> bool:
        """Advanced filtering based on AI strategy"""
        try:
            # Basic sanity checks
            if not compound.get('molecular_weight') or compound['molecular_weight'] <= 0:
                return False
                
            if not compound.get('smiles'):
                return False
            
            # Strategy-based filtering
            scientific_analysis = strategy.get('deep_scientific_analysis', '').lower()
            
            # Filter for electronic materials if relevant
            if any(word in scientific_analysis for word in ['electronic', 'semiconductor', 'conductor']):
                if not self._has_electronic_potential(compound):
                    return False
            
            # Filter for specific functional groups
            target_groups = strategy.get('specific_functional_groups', [])
            if target_groups and not self._has_target_functionality(compound, target_groups):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in advanced filtering: {e}")
            return False

    def _has_electronic_potential(self, compound: Dict) -> bool:
        """Check if compound has electronic material potential"""
        name = compound.get('name', '').lower()
        smiles = compound.get('smiles', '').lower()
        
        electronic_indicators = [
            'conjugated', 'aromatic', 'polymeric', 'fullerene', 'graphene', 
            'nanotube', 'quantum', 'semiconductor', 'conductive', 'photovoltaic',
            'electroluminescent', 'fluorescent', 'phosphorescent'
        ]
        
        return any(indicator in name or indicator in smiles for indicator in electronic_indicators)

    def _has_relevant_functionality(self, compound: Dict, strategy: Dict) -> bool:
        """Check if compound has relevant functional groups"""
        smiles = compound.get('smiles', '').lower()
        target_groups = strategy.get('specific_functional_groups', [])
        
        if not target_groups:
            return True
            
        functional_group_map = {
            'conjugated': ['c=c', 'c#c', 'c1ccccc1'],
            'aromatic': ['c1ccccc1', 'n1ccccc1'],
            'charge_transfer': ['n', 'o', 's'],
            'polar': ['oh', 'c=o', 'n', 'o']
        }
        
        for target_group in target_groups:
            if target_group in functional_group_map:
                patterns = functional_group_map[target_group]
                if any(pattern in smiles for pattern in patterns):
                    return True
        
        return False

    def _has_target_functionality(self, compound: Dict, target_groups: List[str]) -> bool:
        """Check for specific target functional groups"""
        smiles = compound.get('smiles', '').lower()
        
        for group in target_groups:
            if group in smiles:
                return True
        return False

    def _is_innovative(self, compound: Dict) -> bool:
        """Check if compound is innovative/emerging"""
        name = compound.get('name', '').lower()
        
        innovative_indicators = [
            'nano', 'quantum', 'emerging', 'advanced', 'smart', 'functional',
            'hybrid', 'composite', 'multifunctional'
        ]
        
        return any(indicator in name for indicator in innovative_indicators)

    def _remove_duplicates(self, compounds: List[Dict]) -> List[Dict]:
        """Remove duplicate compounds"""
        seen_cids = set()
        unique_compounds = []
        
        for compound in compounds:
            cid = compound.get('cid')
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                unique_compounds.append(compound)
        
        return unique_compounds

    def _ai_enhanced_scoring(self, compounds: List[Dict], strategy: Dict) -> List[Dict]:
        """AI-enhanced scoring of compounds"""
        scored_compounds = []
        
        for compound in compounds:
            score = 0.5  # Base score
            
            # Score based on relevance to strategy
            scientific_analysis = strategy.get('deep_scientific_analysis', '').lower()
            
            # Electronic materials scoring
            if any(word in scientific_analysis for word in ['electronic', 'optical', 'photovoltaic']):
                if self._has_electronic_potential(compound):
                    score += 0.3
            
            # Complexity scoring (more complex molecules often have interesting properties)
            complexity = compound.get('complexity', 0)
            if complexity > 200:
                score += 0.2
            
            # Functional group scoring
            target_groups = strategy.get('specific_functional_groups', [])
            if target_groups and self._has_target_functionality(compound, target_groups):
                score += 0.2
            
            compound['relevance_score'] = min(score, 1.0)
            scored_compounds.append(compound)
        
        # Sort by relevance score
        scored_compounds.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_compounds
