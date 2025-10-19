import pubchempy as pcp
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import random
import time
import logging
import numpy as np

class PubChemManager:
    """
    Intelligent PubChem search manager that uses AI-generated strategies
    to explore the entire chemical space dynamically
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.search_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def intelligent_compound_search(self, scientific_analysis: Dict, search_strategy: Dict, 
                                  material_type: str, max_compounds: int = 200) -> List[Dict]:
        """
        Perform intelligent compound search using AI-generated strategies
        """
        self.logger.info(f"🚀 Starting intelligent search for {material_type}")
        
        all_compounds = []
        
        # Extract search dimensions from strategy
        search_dimensions = search_strategy.get('search_dimensions', {})
        
        # Search across all dimensions
        for dimension_name, search_terms in search_dimensions.items():
            try:
                compounds = self._search_dimension(dimension_name, search_terms, max_compounds // 5)
                all_compounds.extend(compounds)
                self.logger.info(f"✅ {dimension_name} search found {len(compounds)} compounds")
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                self.logger.error(f"❌ {dimension_name} search failed: {e}")
        
        # Remove duplicates and filter
        unique_compounds = self._remove_duplicates(all_compounds)
        filtered_compounds = [c for c in unique_compounds if self._advanced_filter(c, scientific_analysis)]
        
        # Score compounds based on relevance
        scored_compounds = self._score_compounds(filtered_compounds, scientific_analysis)
        
        self.logger.info(f"🎯 Total unique compounds: {len(scored_compounds)}")
        return scored_compounds[:max_compounds]

    def _search_dimension(self, dimension_name: str, search_terms: List[str], max_results: int) -> List[Dict]:
        """Search a specific dimension"""
        compounds = []
        
        for term in search_terms[:10]:  # Limit terms per dimension
            try:
                # Use different search strategies based on dimension
                if dimension_name == 'unconventional':
                    results = self._innovative_search(term, max_results // 2)
                elif dimension_name == 'bio_inspired':
                    results = self._bio_inspired_search(term, max_results // 2)
                else:
                    results = self._standard_search(term, max_results // 3)
                
                compounds.extend(results)
            except Exception as e:
                self.logger.warning(f"Search failed for {term}: {e}")
        
        return compounds

    def _standard_search(self, term: str, max_results: int) -> List[Dict]:
        """Standard PubChem search"""
        try:
            results = pcp.get_compounds(term, 'name', listkey_count=max_results)
            compounds = []
            for compound in results:
                comp_data = self._extract_compound_data(compound)
                if comp_data:
                    compounds.append(comp_data)
            return compounds
        except Exception as e:
            self.logger.error(f"Standard search failed for {term}: {e}")
            return []

    def _innovative_search(self, term: str, max_results: int) -> List[Dict]:
        """Search for innovative compounds"""
        try:
            # Add innovative modifiers to search
            innovative_terms = [f"{term} nanoparticle", f"{term} quantum", f"{term} 2D", 
                              f"{term} composite", f"{term} hybrid"]
            
            compounds = []
            for innovative_term in innovative_terms:
                try:
                    results = pcp.get_compounds(innovative_term, 'name', listkey_count=5)
                    for compound in results:
                        comp_data = self._extract_compound_data(compound)
                        if comp_data and self._is_innovative(comp_data):
                            compounds.append(comp_data)
                except:
                    continue
            
            return compounds
        except Exception as e:
            self.logger.error(f"Innovative search failed for {term}: {e}")
            return []

    def _bio_inspired_search(self, term: str, max_results: int) -> List[Dict]:
        """Search for bio-inspired compounds"""
        try:
            bio_terms = [f"{term} enzyme", f"{term} protein", f"{term} natural", 
                        f"{term} biomimetic", f"{term} bioinspired"]
            
            compounds = []
            for bio_term in bio_terms:
                try:
                    results = pcp.get_compounds(bio_term, 'name', listkey_count=5)
                    for compound in results:
                        comp_data = self._extract_compound_data(compound)
                        if comp_data and self._is_bio_relevant(comp_data):
                            compounds.append(comp_data)
                except:
                    continue
            
            return compounds
        except Exception as e:
            self.logger.error(f"Bio-inspired search failed for {term}: {e}")
            return []

    def _extract_compound_data(self, compound) -> Optional[Dict]:
        """Extract comprehensive compound data from PubChem"""
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

    def _advanced_filter(self, compound: Dict, scientific_analysis: Dict) -> bool:
        """Advanced filtering based on scientific analysis"""
        try:
            # Basic sanity checks
            if not compound.get('molecular_weight') or compound['molecular_weight'] <= 0:
                return False
                
            if not compound.get('smiles'):
                return False
            
            # Strategy-based filtering
            quantum_analysis = scientific_analysis.get('quantum_analysis', '').lower()
            
            # Filter for electronic materials if relevant
            if any(word in quantum_analysis for word in ['electronic', 'semiconductor', 'conductor']):
                if not self._has_electronic_potential(compound):
                    return False
            
            # Filter for specific functional groups
            target_groups = scientific_analysis.get('critical_functional_groups', [])
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

    def _is_innovative(self, compound: Dict) -> bool:
        """Check if compound is innovative/emerging"""
        name = compound.get('name', '').lower()
        
        innovative_indicators = [
            'nano', 'quantum', 'emerging', 'advanced', 'smart', 'functional',
            'hybrid', 'composite', 'multifunctional'
        ]
        
        return any(indicator in name for indicator in innovative_indicators)

    def _is_bio_relevant(self, compound: Dict) -> bool:
        """Check if compound is bio-relevant"""
        name = compound.get('name', '').lower()
        
        bio_indicators = [
            'enzyme', 'protein', 'lipid', 'sugar', 'amino', 'peptide',
            'biological', 'natural', 'plant', 'animal', 'cell'
        ]
        
        return any(indicator in name for indicator in bio_indicators)

    def _has_target_functionality(self, compound: Dict, target_groups: List[str]) -> bool:
        """Check for specific target functional groups"""
        smiles = compound.get('smiles', '').lower()
        name = compound.get('name', '').lower()
        
        for group in target_groups:
            if group in smiles or group in name:
                return True
        return False

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

    def _score_compounds(self, compounds: List[Dict], scientific_analysis: Dict) -> List[Dict]:
        """Score compounds based on relevance to scientific analysis"""
        scored_compounds = []
        
        for compound in compounds:
            score = 0.5  # Base score
            
            # Score based on quantum requirements
            quantum_analysis = scientific_analysis.get('quantum_analysis', '').lower()
            if any(word in quantum_analysis for word in ['electronic', 'optical', 'photovoltaic']):
                if self._has_electronic_potential(compound):
                    score += 0.3
            
            # Score based on molecular complexity
            complexity = compound.get('complexity', 0)
            if complexity > 200:
                score += 0.2
            
            # Score based on functional groups
            target_groups = scientific_analysis.get('critical_functional_groups', [])
            if target_groups and self._has_target_functionality(compound, target_groups):
                score += 0.2
            
            compound['relevance_score'] = min(score, 1.0)
            scored_compounds.append(compound)
        
        # Sort by relevance score
        scored_compounds.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_compounds
