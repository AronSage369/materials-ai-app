import pubchempy as pcp
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
import random
import time
from dataclasses import dataclass
import numpy as np
import re

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
        self.search_cache = {}
        self.compound_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def find_compounds(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Find compounds using PubChem with intelligent search strategies"""
        print(f"🔍 Searching PubChem for {material_type} with strategy: {strategy.get('primary_objective', 'Unknown')}")
        
        # Generate comprehensive search terms
        search_terms = self._generate_intelligent_search_terms(strategy, material_type)
        
        # Try multiple search strategies
        all_compounds = []
        
        # Strategy 1: Direct property-based search
        property_compounds = self._search_by_property_ranges(strategy, material_type)
        all_compounds.extend(property_compounds)
        
        # Strategy 2: Functional group based search
        functional_compounds = self._search_by_functional_groups(strategy, material_type)
        all_compounds.extend(functional_compounds)
        
        # Strategy 3: Traditional name-based search
        name_compounds = self._search_pubchem_comprehensive(search_terms, material_type)
        all_compounds.extend(name_compounds)
        
        # Remove duplicates and filter
        unique_compounds = self._remove_duplicate_compounds(all_compounds)
        filtered_compounds = [c for c in unique_compounds if self._advanced_filter_compound(c, strategy, material_type)]
        
        # Score and categorize
        scored_compounds = self.score_and_categorize_candidates(
            filtered_compounds, strategy.get('target_properties', {})
        )
        
        print(f"✅ Found {len(scored_compounds)} suitable compounds")
        return scored_compounds

    def _generate_intelligent_search_terms(self, strategy: Dict, material_type: str) -> List[str]:
        """Generate comprehensive search terms based on material type and strategy"""
        base_terms = {
            'solvent': [
                'polar aprotic solvent', 'non-polar solvent', 'dipolar solvent', 
                'green solvent', 'ionic liquid', 'deep eutectic solvent',
                'hydrocarbon solvent', 'oxygenated solvent', 'chlorinated solvent',
                'fluorinated solvent', 'siloxane', 'glycol ether', 'ester solvent',
                'ketone solvent', 'alcohol solvent', 'amine solvent', 'sulfoxide',
                'lactone', 'carbonate solvent', 'nitrile solvent'
            ],
            'coolant': [
                'heat transfer fluid', 'thermal oil', 'dielectric fluid', 
                'refrigerant', 'anti-freeze', 'thermal conductive fluid',
                'polyol ester', 'polyalkylene glycol', 'silicone fluid',
                'fluorinated fluid', 'hydrofluoroether', 'mineral oil',
                'synthetic oil', 'glycol mixture', 'brine solution'
            ],
            'absorbent': [
                'porous material', 'adsorbent', 'molecular sieve', 'activated carbon',
                'zeolite', 'silica gel', 'alumina', 'metal organic framework',
                'clay mineral', 'polymer adsorbent', 'ionic exchange resin',
                'mesoporous material', 'hydrogel', 'aerogel'
            ],
            'catalyst': [
                'heterogeneous catalyst', 'homogeneous catalyst', 'enzyme catalyst',
                'acid catalyst', 'base catalyst', 'metal catalyst', 'organocatalyst',
                'zeolite catalyst', 'supported catalyst', 'nanoparticle catalyst',
                'photocatalyst', 'electrocatalyst', 'biocatalyst'
            ],
            'polymer': [
                'thermoplastic', 'thermoset', 'elastomer', 'engineering plastic',
                'biopolymer', 'conductive polymer', 'water soluble polymer',
                'high temperature polymer', 'membrane polymer', 'coating polymer',
                'adhesive polymer', 'composite matrix'
            ]
        }
        
        terms = base_terms.get(material_type.lower(), [material_type])
        
        # Add property-specific terms
        target_props = strategy.get('target_properties', {})
        property_terms = self._get_property_based_terms(target_props)
        terms.extend(property_terms)
        
        # Add application-specific terms from strategy
        primary_obj = strategy.get('primary_objective', '').lower()
        application_terms = self._get_application_terms(primary_obj)
        terms.extend(application_terms)
        
        return list(set(terms))[:15]  # Limit to 15 unique terms

    def _get_property_based_terms(self, target_props: Dict) -> List[str]:
        """Generate search terms based on target properties"""
        terms = []
        
        for prop, criteria in target_props.items():
            if prop == 'thermal_conductivity':
                if criteria.get('min', 0) > 0.5:
                    terms.extend(['high thermal conductivity', 'thermally conductive'])
            elif prop == 'viscosity':
                if criteria.get('max', 100) < 10:
                    terms.extend(['low viscosity', 'mobile liquid'])
                elif criteria.get('min', 0) > 100:
                    terms.extend(['high viscosity', 'viscous liquid'])
            elif prop == 'boiling_point':
                if criteria.get('min', 0) > 200:
                    terms.extend(['high boiling point', 'high bp solvent'])
                elif criteria.get('max', 300) < 100:
                    terms.extend(['low boiling point', 'volatile solvent'])
            elif prop == 'flash_point':
                if criteria.get('min', 0) > 100:
                    terms.extend(['high flash point', 'non-flammable'])
        
        return terms

    def _get_application_terms(self, primary_objective: str) -> List[str]:
        """Extract application-specific terms from primary objective"""
        terms = []
        
        # Common applications and their associated chemical classes
        application_mapping = {
            'electronics': ['dielectric', 'semiconductor', 'conformal coating', 'encapsulant'],
            'automotive': ['lubricant', 'coolant', 'sealant', 'fuel additive'],
            'aerospace': ['high temperature', 'lightweight', 'composite', 'thermal protection'],
            'medical': ['biocompatible', 'sterilizable', 'implant grade', 'medical grade'],
            'energy': ['battery electrolyte', 'fuel cell', 'solar cell', 'energy storage'],
            'coatings': ['protective coating', 'paint base', 'adhesive', 'sealant']
        }
        
        for app, app_terms in application_mapping.items():
            if app in primary_objective.lower():
                terms.extend(app_terms)
        
        return terms

    def _search_by_property_ranges(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Search for compounds based on property ranges"""
        compounds = []
        target_props = strategy.get('target_properties', {})
        
        # Define property-based search strategies
        search_strategies = self._create_property_search_strategies(target_props, material_type)
        
        for strategy_name, search_params in search_strategies.items():
            try:
                # Use PubChem's property filters
                results = pcp.get_compounds(search_params['term'], 'name', 
                                          listkey_count=search_params.get('limit', 10))
                
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data and self._meets_property_criteria(comp_data, target_props):
                        compounds.append(comp_data)
                        
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"Property search failed for {strategy_name}: {e}")
                continue
        
        return compounds

    def _create_property_search_strategies(self, target_props: Dict, material_type: str) -> Dict:
        """Create search strategies based on target properties"""
        strategies = {}
        
        # Molecular weight based searches
        if any(prop in target_props for prop in ['viscosity', 'boiling_point', 'density']):
            strategies['low_mw'] = {'term': 'low molecular weight', 'limit': 15}
            strategies['medium_mw'] = {'term': 'medium molecular weight', 'limit': 15}
        
        # Polarity based searches
        if any(prop in target_props for prop in ['dielectric_constant', 'solubility_parameter']):
            strategies['polar'] = {'term': 'polar compound', 'limit': 15}
            strategies['nonpolar'] = {'term': 'nonpolar compound', 'limit': 15}
        
        # Thermal properties
        if 'thermal_conductivity' in target_props or 'specific_heat' in target_props:
            strategies['thermal_fluid'] = {'term': 'thermal fluid', 'limit': 20}
        
        # Material type specific
        if material_type.lower() == 'solvent':
            strategies['common_solvents'] = {'term': 'common solvent', 'limit': 25}
            strategies['industrial_solvents'] = {'term': 'industrial solvent', 'limit': 20}
        
        return strategies

    def _search_by_functional_groups(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Search for compounds based on functional groups relevant to the application"""
        functional_groups = self._get_relevant_functional_groups(strategy, material_type)
        compounds = []
        
        for fg_group in functional_groups:
            for fg_term in fg_group['terms']:
                try:
                    results = pcp.get_compounds(fg_term, 'name', listkey_count=10)
                    
                    for compound in results:
                        comp_data = self._extract_compound_data(compound)
                        if comp_data:
                            compounds.append(comp_data)
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"Functional group search failed for {fg_term}: {e}")
                    continue
        
        return compounds

    def _get_relevant_functional_groups(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Get functional groups relevant to the material type and strategy"""
        base_groups = {
            'solvent': [
                {'name': 'oxygenated', 'terms': ['alcohol', 'ether', 'ester', 'ketone', 'aldehyde']},
                {'name': 'nitrogenated', 'terms': ['amine', 'amide', 'nitrile', 'nitro compound']},
                {'name': 'halogenated', 'terms': ['chlorinated', 'fluorinated', 'brominated']},
                {'name': 'sulfur', 'terms': ['sulfoxide', 'sulfone', 'sulfonate', 'thiol']}
            ],
            'coolant': [
                {'name': 'glycols', 'terms': ['ethylene glycol', 'propylene glycol', 'polyol']},
                {'name': 'silicon', 'terms': ['siloxane', 'silicone oil', 'polysiloxane']},
                {'name': 'fluorinated', 'terms': ['fluorocarbon', 'hydrofluoroether', 'perfluorinated']}
            ],
            'polymer': [
                {'name': 'thermoplastics', 'terms': ['polyethylene', 'polypropylene', 'polystyrene']},
                {'name': 'engineering', 'terms': ['polycarbonate', 'nylon', 'polyester', 'polyurethane']}
            ]
        }
        
        return base_groups.get(material_type.lower(), [])

    def _search_pubchem_comprehensive(self, search_terms: List[str], material_type: str, 
                                    max_compounds: int = 100) -> List[Dict]:
        """Comprehensive PubChem search with multiple strategies"""
        all_compounds = []
        
        for term in search_terms:
            try:
                print(f"🔍 Searching PubChem for: {term}")
                
                # Try multiple search approaches for each term
                compounds = pcp.get_compounds(term, 'name', listkey_count=15)
                
                for compound in compounds:
                    try:
                        comp_data = self._extract_compound_data(compound)
                        if comp_data:
                            all_compounds.append(comp_data)
                    except Exception as e:
                        print(f"Error processing compound {compound.cid}: {e}")
                        continue
                        
                # Brief delay to be respectful to PubChem
                time.sleep(0.4)
                
            except Exception as e:
                print(f"Error searching for '{term}': {e}")
                continue
        
        # Remove duplicates and limit results
        unique_compounds = self._remove_duplicate_compounds(all_compounds)
        return unique_compounds[:max_compounds]

    def _extract_compound_data(self, compound) -> Optional[Dict]:
        """Extract comprehensive data from PubChem compound"""
        try:
            # Get basic properties
            properties = ['MolecularWeight', 'MolecularFormula', 'CanonicalSMILES', 
                         'IUPACName', 'XLogP', 'Complexity', 'HydrogenBondDonorCount',
                         'HydrogenBondAcceptorCount', 'RotatableBondCount', 'TPSA',
                         'HeavyAtomCount', 'IsotopeAtomCount', 'AtomStereoCount',
                         'DefinedAtomStereoCount', 'UndefinedAtomStereoCount']
            
            comp_props = pcp.get_properties(properties, compound.cid)[0] if compound.cid else {}
            
            # Get additional descriptors
            compound_data = {
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
            
            return compound_data
        except Exception as e:
            print(f"Error extracting data for compound: {e}")
            return None

    def _advanced_filter_compound(self, compound: Dict, strategy: Dict, material_type: str) -> bool:
        """Advanced filtering based on material type and strategy"""
        # Basic sanity checks
        if not compound.get('molecular_weight') or compound['molecular_weight'] <= 0:
            return False
            
        if not compound.get('smiles'):
            return False
        
        mw = compound['molecular_weight']
        
        # Material-specific filtering with wider ranges
        if material_type.lower() == 'solvent':
            # Allow wider range for solvents, including potential solutes
            return 30 <= mw <= 2000
        elif material_type.lower() == 'coolant':
            return 50 <= mw <= 5000
        elif material_type.lower() == 'polymer':
            return mw >= 500  # Lower threshold for oligomers
        elif material_type.lower() == 'catalyst':
            return 50 <= mw <= 5000
        elif material_type.lower() == 'absorbent':
            return 50 <= mw <= 10000
        else:
            return 50 <= mw <= 5000  # General range

    def _meets_property_criteria(self, compound: Dict, target_props: Dict) -> bool:
        """Check if compound meets basic property criteria"""
        # Simple heuristic checks based on molecular properties
        mw = compound.get('molecular_weight', 0)
        logp = compound.get('logp', 0)
        
        for prop, criteria in target_props.items():
            if prop == 'viscosity':
                # Lower MW generally means lower viscosity
                min_viscosity = criteria.get('min', 0)
                if min_viscosity > 10 and mw < 100:
                    return False
            elif prop == 'boiling_point':
                min_bp = criteria.get('min', 0)
                if min_bp > 200 and mw < 100:
                    return False
            elif prop == 'solubility_parameter':
                # LogP can indicate polarity
                target_polarity = criteria.get('target')
                if target_polarity and abs(logp - (5 - target_polarity/3)) > 3:
                    return False
        
        return True

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
        """Advanced scoring and categorization of compounds"""
        scored_candidates = []
        
        for candidate in candidates:
            candidate['scores'] = {}
            total_score = 0
            property_count = 0
            
            # Score based on multiple criteria
            scoring_functions = [
                self._score_by_molecular_weight,
                self._score_by_complexity,
                self._score_by_polarity,
                self._score_by_functional_groups
            ]
            
            for score_func in scoring_functions:
                score = score_func(candidate, target_properties)
                if score > 0:
                    func_name = score_func.__name__.replace('_score_by_', '')
                    candidate['scores'][func_name] = score
                    total_score += score
                    property_count += 1
                
            # Calculate overall score
            candidate['overall_score'] = total_score / property_count if property_count > 0 else 0.5
            
            # Advanced categorization
            candidate['category'] = self._advanced_categorize_compound(candidate, target_properties)
            scored_candidates.append(candidate)
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        return scored_candidates

    def _score_by_molecular_weight(self, candidate: Dict, target_properties: Dict) -> float:
        """Advanced molecular weight scoring"""
        mw = candidate.get('molecular_weight', 0)
        if not mw or mw <= 0:
            return 0.5
            
        # Property-specific MW optima with wider acceptance
        if any(prop in target_properties for prop in ['viscosity', 'diffusion_coefficient']):
            # Lower MW generally better for low viscosity/fast diffusion
            return max(0, 1 - (mw - 50) / 1000)
        elif any(prop in target_properties for prop in ['thermal_stability', 'boiling_point']):
            # Higher MW generally better for thermal properties
            return min(1, mw / 800)
        else:
            # Balanced approach with wider peak
            return 1 - abs(mw - 250) / 500

    def _score_by_complexity(self, candidate: Dict, target_properties: Dict) -> float:
        """Score based on molecular complexity"""
        complexity = candidate.get('complexity', 0)
        if not complexity or complexity <= 0:
            return 0.5
            
        normalized_complexity = min(complexity / 600, 1.0)
        
        # Adjust based on application needs
        if any(prop in target_properties for prop in ['cost', 'synthesis', 'availability']):
            return 1 - normalized_complexity * 0.6
        elif any(prop in target_properties for prop in ['specificity', 'selectivity']):
            return 0.3 + normalized_complexity * 0.7
        else:
            return 0.5 + normalized_complexity * 0.2

    def _score_by_polarity(self, candidate: Dict, target_properties: Dict) -> float:
        """Score based on polarity indicators"""
        logp = candidate.get('logp', 0)
        tpsa = candidate.get('tpsa', 0)
        hbd = candidate.get('hbd_count', 0)
        hba = candidate.get('hba_count', 0)
        
        # Calculate polarity score
        polarity_score = 0.0
        
        # LogP based polarity (lower LogP = more polar)
        if logp is not None:
            polarity_score += (5 - min(max(logp, -5), 5)) / 10
        
        # TPSA based polarity
        if tpsa is not None:
            polarity_score += min(tpsa / 200, 1.0) / 2
        
        # Hydrogen bonding
        hbond_score = min((hbd + hba) / 8, 1.0) / 3
        polarity_score += hbond_score
        
        # Normalize
        polarity_score = min(polarity_score, 1.0)
        
        # Adjust based on target properties
        if any(prop in target_properties for prop in ['dielectric_constant', 'solubility_parameter']):
            return polarity_score
        elif any(prop in target_properties for prop in ['hydrophobicity', 'logp']):
            return 1 - polarity_score
        else:
            return 0.7  # Neutral preference

    def _score_by_functional_groups(self, candidate: Dict, target_properties: Dict) -> float:
        """Score based on functional groups relevant to application"""
        smiles = candidate.get('smiles', '').lower()
        name = candidate.get('name', '').lower()
        
        score = 0.5  # Base score
        
        # Functional groups that are generally desirable
        desirable_groups = [
            'oh', 'cooh', 'c=o', 'o', 'n', 's=o', 'cn', 
            'cooc', 'coc', 'cc', 'cl', 'f', 'br'
        ]
        
        # Count desirable groups
        group_count = sum(1 for group in desirable_groups if group in smiles)
        score += min(group_count * 0.1, 0.3)
        
        return min(score, 1.0)

    def _advanced_categorize_compound(self, candidate: Dict, target_properties: Dict) -> str:
        """Advanced categorization based on multiple properties"""
        scores = candidate.get('scores', {})
        
        if len(scores) >= 2:
            score_values = list(scores.values())
            avg_score = np.mean(score_values)
            std_score = np.std(score_values)
            
            if std_score < 0.15 and avg_score > 0.6:
                return 'balanced'
            elif any(score > 0.8 for score in score_values):
                return 'specialist'
            else:
                return 'general'
        else:
            return 'general'
