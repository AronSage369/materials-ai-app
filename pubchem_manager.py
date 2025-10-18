import pubchempy as pcp
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
import random
import time
import logging
from dataclasses import dataclass
import numpy as np
import re
from utils import cached, CacheManager, MemoryManager

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
        self.logger = logging.getLogger(__name__)
        self.cache = CacheManager()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @cached
    def find_compounds(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Find compounds using comprehensive PubChem search across ALL chemical classes"""
        self.logger.info(f"🔍 Comprehensive PubChem search for {material_type}")
        
        try:
            # Generate MASSIVE search terms covering all chemical classes
            search_terms = self._generate_comprehensive_search_terms(strategy, material_type)
            
            # Use multiple search strategies
            all_compounds = []
            
            # Strategy 1: Broad category searches
            category_compounds = self._search_by_categories(strategy, material_type)
            if category_compounds:
                all_compounds.extend(category_compounds)
            
            # Strategy 2: Property-based searches
            property_compounds = self._search_by_properties(strategy, material_type)
            if property_compounds:
                all_compounds.extend(property_compounds)
            
            # Strategy 3: Comprehensive term-based search
            term_compounds = self._search_pubchem_comprehensive(search_terms, material_type, max_compounds=200)
            if term_compounds:
                all_compounds.extend(term_compounds)
            
            # Strategy 4: Direct compound class searches
            class_compounds = self._search_compound_classes(strategy, material_type)
            if class_compounds:
                all_compounds.extend(class_compounds)
            
            # If still few results, use enhanced fallback
            if len(all_compounds) < 20:
                self.logger.warning("Few compounds found, expanding search...")
                expanded_compounds = self._expanded_fallback_search(material_type)
                all_compounds.extend(expanded_compounds)
            
            # Remove duplicates and apply minimal filtering
            unique_compounds = self._remove_duplicate_compounds(all_compounds)
            filtered_compounds = [c for c in unique_compounds if self._minimal_filter_compound(c, material_type)]
            
            # Enhanced scoring and categorization
            scored_compounds = self.enhanced_score_and_categorize(filtered_compounds, strategy)
            
            self.logger.info(f"✅ Found {len(scored_compounds)} diverse compounds from {len(unique_compounds)} total")
            return scored_compounds[:100]  # Return top 100
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive search: {e}")
            return self._get_enhanced_fallback_compounds(material_type)

    def _generate_comprehensive_search_terms(self, strategy: Dict, material_type: str) -> List[str]:
        """Generate massive list of search terms covering ALL chemical classes"""
        terms = []
        
        # Base material type terms
        base_terms = self._get_base_material_terms(material_type)
        terms.extend(base_terms)
        
        # Add ALL chemical class terms
        chemical_classes = self._get_all_chemical_classes()
        terms.extend(chemical_classes)
        
        # Add functional group terms
        functional_terms = self._get_functional_group_terms()
        terms.extend(functional_terms)
        
        # Add application-specific terms
        application_terms = self._get_application_terms(strategy)
        terms.extend(application_terms)
        
        # Add property-based terms
        property_terms = self._get_enhanced_property_terms(strategy)
        terms.extend(property_terms)
        
        # Add specific compound examples
        specific_compounds = self._get_specific_compound_examples()
        terms.extend(specific_compounds)
        
        # Remove duplicates and limit
        unique_terms = list(set([str(term).lower() for term in terms if term]))
        return unique_terms[:50]  # Limit to 50 most relevant terms

    def _get_base_material_terms(self, material_type: str) -> List[str]:
        """Get base terms for the material type"""
        material_terms = {
            'solvent': [
                'solvent', 'polar solvent', 'non-polar solvent', 'aprotic solvent', 
                'protic solvent', 'organic solvent', 'inorganic solvent', 'green solvent',
                'industrial solvent', 'laboratory solvent', 'extraction solvent',
                'reaction solvent', 'cleaning solvent', 'degreasing solvent'
            ],
            'coolant': [
                'coolant', 'heat transfer fluid', 'thermal fluid', 'refrigerant',
                'anti-freeze', 'thermal oil', 'dielectric coolant', 'liquid coolant',
                'electronic coolant', 'engine coolant', 'industrial coolant'
            ],
            'absorbent': [
                'absorbent', 'adsorbent', 'porous material', 'molecular sieve',
                'desiccant', 'drying agent', 'moisture absorber', 'gas absorber',
                'liquid absorber', 'spill absorber', 'industrial absorbent'
            ],
            'catalyst': [
                'catalyst', 'catalytic', 'enzyme', 'biocatalyst', 'heterogeneous catalyst',
                'homogeneous catalyst', 'photocatalyst', 'electrocatalyst', 'reaction catalyst',
                'industrial catalyst', 'chemical catalyst'
            ],
            'polymer': [
                'polymer', 'plastic', 'resin', 'elastomer', 'thermoplastic',
                'thermoset', 'copolymer', 'biopolymer', 'engineering plastic',
                'polymer matrix', 'composite polymer'
            ],
            'lubricant': [
                'lubricant', 'lubricating oil', 'grease', 'lubricating grease',
                'industrial lubricant', 'engine oil', 'machine oil', 'synthetic lubricant'
            ]
        }
        
        return material_terms.get(material_type.lower(), [material_type])

    def _get_all_chemical_classes(self) -> List[str]:
        """Get comprehensive list of ALL chemical classes"""
        return [
            # Organic Compounds
            'hydrocarbon', 'alkane', 'alkene', 'alkyne', 'aromatic', 'arene',
            'alcohol', 'ether', 'aldehyde', 'ketone', 'carboxylic acid', 'ester',
            'phenol', 'amine', 'amide', 'nitrile', 'nitro compound', 'halogenated',
            'organosulfur', 'organophosphorus', 'organometallic',
            
            # Fatty acids and lipids
            'fatty acid', 'lipid', 'triglyceride', 'phospholipid', 'steroid',
            'oleic acid', 'stearic acid', 'palmitic acid', 'linoleic acid',
            'arachidonic acid', 'omega-3', 'omega-6', 'saturated fat', 'unsaturated fat',
            
            # Bio-based compounds
            'bio oil', 'vegetable oil', 'plant extract', 'essential oil', 'enzyme',
            'protein', 'peptide', 'amino acid', 'carbohydrate', 'sugar', 'starch',
            'cellulose', 'lignin', 'chitosan', 'alginate',
            
            # Inorganic compounds
            'inorganic', 'mineral', 'salt', 'oxide', 'hydroxide', 'acid', 'base',
            'coordination compound', 'complex', 'metal complex', 'transition metal',
            'alkali metal', 'alkaline earth', 'lanthanide', 'actinide',
            
            # Surfactants and emulsifiers
            'surfactant', 'emulsifier', 'detergent', 'soap', 'wetting agent',
            'anionic surfactant', 'cationic surfactant', 'nonionic surfactant',
            'zwitterionic surfactant',
            
            # Polymers
            'polymer', 'polyethylene', 'polypropylene', 'polystyrene', 'pvc',
            'nylon', 'polyester', 'polyurethane', 'epoxy', 'silicone', 'rubber',
            'elastomer', 'hydrogel', 'aerogel',
            
            # Advanced materials
            'nanomaterial', 'nanoparticle', 'quantum dot', 'carbon nanotube',
            'graphene', 'fullerene', 'metal organic framework', 'mof',
            'zeolite', 'mesoporous material', 'composite', 'ceramic',
            
            # Energy materials
            'battery material', 'fuel cell', 'solar cell', 'thermoelectric',
            'supercapacitor', 'energy storage', 'conductor', 'semiconductor',
            
            # Deuterated and specialty compounds
            'deuterated', 'heavy water', 'd2o', 'isotope', 'labeled compound',
            'fluorescent', 'luminescent', 'photochromic', 'electrochromic',
            
            # Industrial chemicals
            'industrial chemical', 'commodity chemical', 'fine chemical',
            'specialty chemical', 'bulk chemical', 'intermediate'
        ]

    def _get_functional_group_terms(self) -> List[str]:
        """Get terms for functional groups"""
        return [
            'hydroxyl', 'carbonyl', 'carboxyl', 'amino', 'imino', 'nitro',
            'sulfhydryl', 'disulfide', 'sulfoxide', 'sulfone', 'phosphate',
            'phosphonate', 'halogen', 'fluoro', 'chloro', 'bromo', 'iodo',
            'cyano', 'isocyano', 'azo', 'diazo', 'epoxy', 'peroxy'
        ]

    def _get_application_terms(self, strategy: Dict) -> List[str]:
        """Get application-specific terms from strategy"""
        terms = []
        scientific_analysis = strategy.get('scientific_analysis', '').lower()
        
        application_keywords = {
            'electronics': ['dielectric', 'semiconductor', 'electronic', 'conformal', 'encapsulant'],
            'automotive': ['automotive', 'engine', 'fuel', 'transmission', 'brake'],
            'aerospace': ['aerospace', 'aviation', 'rocket', 'satellite', 'high temperature'],
            'medical': ['medical', 'biocompatible', 'implant', 'drug delivery', 'tissue engineering'],
            'energy': ['energy', 'battery', 'solar', 'fuel cell', 'storage'],
            'coatings': ['coating', 'paint', 'adhesive', 'sealant', 'protective'],
            'environmental': ['environmental', 'green', 'sustainable', 'biodegradable', 'renewable'],
            'industrial': ['industrial', 'manufacturing', 'process', 'production']
        }
        
        for app, keywords in application_keywords.items():
            if any(keyword in scientific_analysis for keyword in [app] + keywords):
                terms.extend(keywords)
        
        return terms

    def _get_enhanced_property_terms(self, strategy: Dict) -> List[str]:
        """Get property-based search terms"""
        terms = []
        target_props = strategy.get('target_properties', {})
        
        property_mapping = {
            'thermal_conductivity': ['thermally conductive', 'heat conductor', 'thermal conductor'],
            'viscosity': ['low viscosity', 'high viscosity', 'mobile liquid', 'viscous'],
            'boiling_point': ['high boiling', 'low boiling', 'volatile', 'non-volatile'],
            'flash_point': ['high flash point', 'non-flammable', 'flammable'],
            'density': ['low density', 'high density', 'lightweight', 'heavy'],
            'surface_tension': ['low surface tension', 'high surface tension', 'wetting'],
            'solubility': ['soluble', 'insoluble', 'hydrophilic', 'hydrophobic'],
            'stability': ['stable', 'thermal stability', 'chemical stability']
        }
        
        for prop in target_props.keys():
            if prop in property_mapping:
                terms.extend(property_mapping[prop])
        
        return terms

    def _get_specific_compound_examples(self) -> List[str]:
        """Get specific compound examples for diverse searching"""
        return [
            # Common solvents and chemicals
            'water', 'methanol', 'ethanol', 'isopropanol', 'acetone', 'acetonitrile',
            'dichloromethane', 'chloroform', 'tetrahydrofuran', 'dimethyl sulfoxide',
            'dimethylformamide', 'hexane', 'toluene', 'diethyl ether', 'ethyl acetate',
            
            # Fatty acids and bio compounds
            'oleic acid', 'stearic acid', 'palmitic acid', 'linoleic acid', 'glycerol',
            'lecithin', 'cholesterol', 'vitamin e', 'squalene',
            
            # Inorganics and minerals
            'sodium chloride', 'calcium carbonate', 'silica', 'alumina', 'titanium dioxide',
            'zinc oxide', 'iron oxide', 'copper sulfate', 'sodium hydroxide',
            
            # Polymers and materials
            'polyethylene glycol', 'polystyrene', 'nylon-6', 'polycarbonate', 'silicone oil',
            'polydimethylsiloxane', 'polyvinyl alcohol', 'polyacrylic acid',
            
            # Surfactants
            'sodium dodecyl sulfate', 'tween', 'triton', 'span', 'brij',
            
            # Special compounds
            'deuterium oxide', 'heavy water', 'ionic liquid', 'deep eutectic solvent'
        ]

    def _search_by_categories(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Search by broad chemical categories"""
        compounds = []
        categories = self._get_search_categories(material_type)
        
        for category in categories[:10]:  # Limit to 10 categories
            try:
                self.logger.info(f"🔍 Searching category: {category}")
                results = pcp.get_compounds(category, 'name', listkey_count=20)
                
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        compounds.append(comp_data)
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"Category search failed for {category}: {e}")
                continue
        
        return compounds

    def _get_search_categories(self, material_type: str) -> List[str]:
        """Get search categories based on material type"""
        category_map = {
            'solvent': [
                'organic solvent', 'polar solvent', 'aprotic solvent', 'ionic liquid',
                'green solvent', 'fluorinated solvent', 'chlorinated solvent'
            ],
            'coolant': [
                'heat transfer fluid', 'refrigerant', 'thermal oil', 'dielectric fluid',
                'anti-freeze', 'silicone fluid', 'fluorinated coolant'
            ],
            'absorbent': [
                'porous material', 'adsorbent', 'molecular sieve', 'activated carbon',
                'zeolite', 'silica gel', 'metal organic framework'
            ],
            'catalyst': [
                'heterogeneous catalyst', 'homogeneous catalyst', 'enzyme', 'photocatalyst',
                'electrocatalyst', 'metal catalyst', 'acid catalyst'
            ],
            'polymer': [
                'thermoplastic', 'thermoset', 'elastomer', 'engineering plastic',
                'biopolymer', 'conductive polymer', 'water soluble polymer'
            ],
            'lubricant': [
                'lubricating oil', 'synthetic lubricant', 'grease', 'solid lubricant',
                'industrial lubricant', 'engine oil'
            ]
        }
        
        return category_map.get(material_type.lower(), ['chemical compound'])

    def _search_by_properties(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Search based on target properties"""
        compounds = []
        property_terms = self._generate_property_search_terms(strategy)
        
        for term in property_terms[:5]:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=15)
                
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        compounds.append(comp_data)
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"Property search failed for {term}: {e}")
                continue
        
        return compounds

    def _generate_property_search_terms(self, strategy: Dict) -> List[str]:
        """Generate search terms from target properties"""
        terms = []
        target_props = strategy.get('target_properties', {})
        
        for prop, criteria in target_props.items():
            if not isinstance(criteria, dict):
                continue
                
            if prop == 'thermal_conductivity':
                min_val = criteria.get('min')
                if isinstance(min_val, (int, float)) and min_val > 0.5:
                    terms.append('high thermal conductivity')
            elif prop == 'viscosity':
                max_val = criteria.get('max')
                if isinstance(max_val, (int, float)) and max_val < 10:
                    terms.append('low viscosity')
                elif isinstance(max_val, (int, float)) and max_val > 100:
                    terms.append('high viscosity')
        
        return terms

    def _search_compound_classes(self, strategy: Dict, material_type: str) -> List[Dict]:
        """Search specific compound classes"""
        compounds = []
        classes_to_search = self._get_compound_classes_for_material(material_type)
        
        for compound_class in classes_to_search[:8]:
            try:
                results = pcp.get_compounds(compound_class, 'name', listkey_count=12)
                
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        compounds.append(comp_data)
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"Class search failed for {compound_class}: {e}")
                continue
        
        return compounds

    def _get_compound_classes_for_material(self, material_type: str) -> List[str]:
        """Get specific compound classes for material type"""
        class_map = {
            'solvent': [
                'alcohol', 'ketone', 'ester', 'ether', 'sulfoxide', 'nitrile',
                'halogenated solvent', 'ionic liquid', 'deep eutectic solvent'
            ],
            'coolant': [
                'glycol', 'silicone', 'fluorocarbon', 'mineral oil', 'synthetic oil',
                'polyol ester', 'hydrofluoroether'
            ],
            'absorbent': [
                'activated carbon', 'zeolite', 'silica', 'alumina', 'clay',
                'polymer adsorbent', 'metal organic framework'
            ],
            'catalyst': [
                'metal catalyst', 'enzyme', 'zeolite catalyst', 'acid catalyst',
                'base catalyst', 'nanoparticle catalyst', 'organocatalyst'
            ],
            'polymer': [
                'polyolefin', 'polyester', 'polyamide', 'polyurethane', 'epoxy',
                'silicone', 'biodegradable polymer', 'conductive polymer'
            ],
            'lubricant': [
                'mineral oil', 'synthetic oil', 'grease', 'solid lubricant',
                'biodegradable lubricant', 'high temperature lubricant'
            ]
        }
        
        return class_map.get(material_type.lower(), ['organic compound', 'inorganic compound'])

    def _search_pubchem_comprehensive(self, search_terms: List[str], material_type: str, 
                                    max_compounds: int = 150) -> List[Dict]:
        """Comprehensive PubChem search with massive term coverage"""
        all_compounds = []
        
        for term in search_terms:
            try:
                if not isinstance(term, str):
                    continue
                    
                self.logger.info(f"🔍 Searching: {term}")
                compounds = pcp.get_compounds(term, 'name', listkey_count=10)
                
                for compound in compounds:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        all_compounds.append(comp_data)
                
                time.sleep(0.4)
                
            except Exception as e:
                self.logger.error(f"Search failed for '{term}': {e}")
                continue
        
        # Remove duplicates
        unique_compounds = self._remove_duplicate_compounds(all_compounds)
        return unique_compounds[:max_compounds]

    def _expanded_fallback_search(self, material_type: str) -> List[Dict]:
        """Expanded fallback search when primary searches fail"""
        compounds = []
        fallback_terms = self._get_expanded_fallback_terms(material_type)
        
        for term in fallback_terms:
            try:
                results = pcp.get_compounds(term, 'name', listkey_count=8)
                
                for compound in results:
                    comp_data = self._extract_compound_data(compound)
                    if comp_data:
                        compounds.append(comp_data)
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"Fallback search failed for {term}: {e}")
                continue
        
        return compounds

    def _get_expanded_fallback_terms(self, material_type: str) -> List[str]:
        """Get expanded fallback search terms"""
        return [
            'chemical', 'compound', 'organic', 'inorganic', 'industrial chemical',
            'commercial chemical', 'laboratory chemical', 'research chemical'
        ]

    def _extract_compound_data(self, compound) -> Optional[Dict]:
        """Extract comprehensive data from PubChem compound"""
        try:
            properties = ['MolecularWeight', 'MolecularFormula', 'CanonicalSMILES', 
                         'IUPACName', 'XLogP', 'Complexity', 'HydrogenBondDonorCount',
                         'HydrogenBondAcceptorCount', 'RotatableBondCount', 'TPSA']
            
            comp_props = pcp.get_properties(properties, compound.cid)[0] if compound.cid else {}
            
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
            self.logger.error(f"Error extracting data for compound: {e}")
            return None

    def _minimal_filter_compound(self, compound: Dict, material_type: str) -> bool:
        """Minimal filtering - only remove completely invalid compounds"""
        try:
            # Basic sanity checks
            if not compound.get('molecular_weight') or compound['molecular_weight'] <= 0:
                return False
                
            if not compound.get('smiles'):
                return False
            
            # Very permissive molecular weight ranges
            mw = compound['molecular_weight']
            return 10 <= mw <= 50000  # Allow very small to very large molecules
            
        except Exception as e:
            self.logger.error(f"Error in minimal filtering: {e}")
            return False

    def _remove_duplicate_compounds(self, compounds: List[Dict]) -> List[Dict]:
        """Remove duplicate compounds based on CID"""
        try:
            seen_cids = set()
            unique_compounds = []
            
            for compound in compounds:
                cid = compound.get('cid')
                if cid and cid not in seen_cids:
                    seen_cids.add(cid)
                    unique_compounds.append(compound)
                    
            return unique_compounds
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {e}")
            return compounds

    def enhanced_score_and_categorize(self, compounds: List[Dict], strategy: Dict) -> List[Dict]:
        """Enhanced scoring that considers chemical diversity"""
        try:
            scored_compounds = []
            target_props = strategy.get('target_properties', {})
            
            for compound in compounds:
                compound['scores'] = {}
                total_score = 0
                criteria_count = 0
                
                # Score based on multiple criteria
                scoring_methods = [
                    self._score_molecular_properties,
                    self._score_chemical_class,
                    self._score_functional_groups,
                    self._score_application_fit
                ]
                
                for score_method in scoring_methods:
                    score = score_method(compound, target_props, strategy)
                    if score > 0:
                        method_name = score_method.__name__.replace('_score_', '')
                        compound['scores'][method_name] = score
                        total_score += score
                        criteria_count += 1
                
                # Calculate overall score
                compound['overall_score'] = total_score / criteria_count if criteria_count > 0 else 0.5
                
                # Enhanced categorization
                compound['category'] = self._enhanced_categorization(compound, strategy)
                scored_compounds.append(compound)
            
            # Sort by overall score and ensure diversity
            scored_compounds.sort(key=lambda x: x['overall_score'], reverse=True)
            return self._ensure_diversity(scored_compounds)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced scoring: {e}")
            return compounds

    def _score_molecular_properties(self, compound: Dict, target_props: Dict, strategy: Dict) -> float:
        """Score based on molecular properties"""
        try:
            mw = compound.get('molecular_weight', 0)
            logp = compound.get('logp', 0)
            tpsa = compound.get('tpsa', 0)
            
            score = 0.5  # Base score
            
            # Adjust based on target properties
            for prop, criteria in target_props.items():
                if not isinstance(criteria, dict):
                    continue
                    
                if prop == 'viscosity':
                    # Lower MW generally means lower viscosity
                    min_visc = criteria.get('min')
                    if isinstance(min_visc, (int, float)) and min_visc > 10:
                        score += (1 - min(mw / 1000, 1)) * 0.3
                
                elif prop == 'thermal_conductivity':
                    # Some correlation with molecular simplicity
                    min_thermal = criteria.get('min')
                    if isinstance(min_thermal, (int, float)) and min_thermal > 0.3:
                        complexity = compound.get('complexity', 0)
                        score += (1 - min(complexity / 500, 1)) * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in molecular property scoring: {e}")
            return 0.5

    def _score_chemical_class(self, compound: Dict, target_props: Dict, strategy: Dict) -> float:
        """Score based on chemical class relevance"""
        try:
            name = compound.get('name', '').lower()
            smiles = compound.get('smiles', '').lower()
            
            score = 0.5
            
            # Score based on chemical class indicators
            class_indicators = {
                'fatty acid': 0.8, 'lipid': 0.7, 'polymer': 0.6, 'surfactant': 0.7,
                'enzyme': 0.9, 'protein': 0.8, 'nanomaterial': 0.8, 'composite': 0.7,
                'deuterated': 0.6, 'ionic liquid': 0.7, 'deep eutectic': 0.7
            }
            
            for indicator, boost in class_indicators.items():
                if indicator in name or indicator in smiles:
                    score = max(score, boost)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in chemical class scoring: {e}")
            return 0.5

    def _score_functional_groups(self, compound: Dict, target_props: Dict, strategy: Dict) -> float:
        """Score based on functional groups"""
        try:
            smiles = compound.get('smiles', '').lower()
            
            # Common functional groups
            functional_groups = [
                'oh', 'cooh', 'c=o', 'o', 'n', 's=o', 'cn', 'no2',
                'cl', 'f', 'br', 'i', 'p', 's'
            ]
            
            # Count diverse functional groups
            unique_groups = sum(1 for group in functional_groups if group in smiles)
            score = 0.3 + (min(unique_groups, 5) * 0.1)
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in functional group scoring: {e}")
            return 0.5

    def _score_application_fit(self, compound: Dict, target_props: Dict, strategy: Dict) -> float:
        """Score based on application fit"""
        try:
            scientific_analysis = strategy.get('scientific_analysis', '').lower()
            name = compound.get('name', '').lower()
            
            score = 0.5
            
            # Application-specific scoring
            if 'electronic' in scientific_analysis:
                if any(keyword in name for keyword in ['dielectric', 'insulating', 'semiconductor']):
                    score = 0.8
            elif 'medical' in scientific_analysis:
                if any(keyword in name for keyword in ['biocompatible', 'medical', 'pharmaceutical']):
                    score = 0.8
            elif 'energy' in scientific_analysis:
                if any(keyword in name for keyword in ['battery', 'fuel cell', 'solar', 'energy']):
                    score = 0.8
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in application fit scoring: {e}")
            return 0.5

    def _enhanced_categorization(self, compound: Dict, strategy: Dict) -> str:
        """Enhanced categorization considering chemical diversity"""
        try:
            scores = compound.get('scores', {})
            
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
                
        except Exception as e:
            self.logger.error(f"Error in enhanced categorization: {e}")
            return 'general'

    def _ensure_diversity(self, compounds: List[Dict]) -> List[Dict]:
        """Ensure chemical diversity in the results"""
        try:
            if len(compounds) <= 10:
                return compounds
            
            # Group by chemical class indicators
            diverse_compounds = []
            classes_covered = set()
            
            for compound in compounds:
                compound_class = self._classify_compound(compound)
                
                if compound_class not in classes_covered or len(diverse_compounds) < 5:
                    diverse_compounds.append(compound)
                    classes_covered.add(compound_class)
                
                if len(diverse_compounds) >= 50:  # Ensure we get diverse compounds
                    break
            
            # Add some high-scoring compounds regardless of class
            remaining_slots = 50 - len(diverse_compounds)
            if remaining_slots > 0:
                for compound in compounds:
                    if compound not in diverse_compounds:
                        diverse_compounds.append(compound)
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
            
            return diverse_compounds
            
        except Exception as e:
            self.logger.error(f"Error ensuring diversity: {e}")
            return compounds[:50]

    def _classify_compound(self, compound: Dict) -> str:
        """Classify compound into broad chemical classes"""
        try:
            name = compound.get('name', '').lower()
            smiles = compound.get('smiles', '').lower()
            
            # Check for class indicators
            if any(indicator in name or indicator in smiles 
                   for indicator in ['acid', 'carboxyl']):
                return 'acid'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['alcohol', 'hydroxyl']):
                return 'alcohol'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['amine', 'amino']):
                return 'amine'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['ester', 'coo']):
                return 'ester'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['polymer', 'poly']):
                return 'polymer'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['salt', 'chloride', 'sulfate']):
                return 'salt'
            elif any(indicator in name or indicator in smiles 
                   for indicator in ['hydrocarbon', 'ane', 'ene', 'yne']):
                return 'hydrocarbon'
            else:
                return 'other'
                
        except Exception as e:
            self.logger.error(f"Error in compound classification: {e}")
            return 'unknown'

    def _get_enhanced_fallback_compounds(self, material_type: str) -> List[Dict]:
        """Get enhanced fallback compounds covering diverse chemical classes"""
        try:
            # Extensive fallback database with diverse compounds
            fallback_compounds = {
                'solvent': [
                    {'cid': 887, 'name': 'Methanol', 'molecular_weight': 32.04, 'smiles': 'CO', 'category': 'balanced'},
                    {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                    {'cid': 6344, 'name': 'Ethanol', 'molecular_weight': 46.07, 'smiles': 'CCO', 'category': 'balanced'},
                    {'cid': 6579, 'name': 'Acetone', 'molecular_weight': 58.08, 'smiles': 'CC(=O)C', 'category': 'balanced'},
                    {'cid': 7507, 'name': 'Dimethyl Sulfoxide', 'molecular_weight': 78.13, 'smiles': 'CS(C)=O', 'category': 'specialist'},
                    {'cid': 6212, 'name': 'Hexane', 'molecular_weight': 86.18, 'smiles': 'CCCCCC', 'category': 'balanced'},
                    {'cid': 1140, 'name': 'Diethyl Ether', 'molecular_weight': 74.12, 'smiles': 'CCOCC', 'category': 'balanced'},
                    {'cid': 8857, 'name': 'Ethyl Acetate', 'molecular_weight': 88.11, 'smiles': 'CCOC(C)=O', 'category': 'balanced'},
                    {'cid': 7964, 'name': 'Tetrahydrofuran', 'molecular_weight': 72.11, 'smiles': 'C1CCOC1', 'category': 'balanced'},
                    {'cid': 679, 'name': 'Dimethylformamide', 'molecular_weight': 73.09, 'smiles': 'CN(C)C=O', 'category': 'specialist'},
                    # Fatty acids and bio compounds
                    {'cid': 445639, 'name': 'Oleic Acid', 'molecular_weight': 282.47, 'smiles': 'CCCCCCCC=CCCCCCCCC(=O)O', 'category': 'specialist'},
                    {'cid': 5281, 'name': 'Stearic Acid', 'molecular_weight': 284.48, 'smiles': 'CCCCCCCCCCCCCCCCCC(=O)O', 'category': 'specialist'},
                    {'cid': 985, 'name': 'Palmitic Acid', 'molecular_weight': 256.43, 'smiles': 'CCCCCCCCCCCCCCCC(=O)O', 'category': 'specialist'},
                    # Ionic liquids and special compounds
                    {'cid': 2734162, 'name': '1-Butyl-3-methylimidazolium chloride', 'molecular_weight': 174.67, 'smiles': 'CCCC[n+]1c[nH]c(c1)C.[Cl-]', 'category': 'specialist'},
                    {'cid': 24639, 'name': 'Heavy Water', 'molecular_weight': 20.03, 'smiles': '[2H]O[2H]', 'category': 'specialist'}
                ],
                'coolant': [
                    {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                    {'cid': 174, 'name': 'Ethylene Glycol', 'molecular_weight': 62.07, 'smiles': 'OCCO', 'category': 'specialist'},
                    {'cid': 1030, 'name': 'Propylene Glycol', 'molecular_weight': 76.09, 'smiles': 'CC(O)CO', 'category': 'specialist'},
                    {'cid': 753, 'name': 'Glycerol', 'molecular_weight': 92.09, 'smiles': 'C(C(CO)O)O', 'category': 'balanced'},
                    {'cid': 6436606, 'name': 'Silicone Oil', 'molecular_weight': 500.0, 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'category': 'specialist'},
                    {'cid': 6386, 'name': 'Mineral Oil', 'molecular_weight': 400.0, 'smiles': 'CCCCCCCCCCCCCCCC', 'category': 'balanced'}
                ],
                'polymer': [
                    {'cid': 6325, 'name': 'Polyethylene', 'molecular_weight': 28000, 'smiles': 'C=C', 'category': 'balanced'},
                    {'cid': 76958, 'name': 'Polypropylene', 'molecular_weight': 42000, 'smiles': 'CC=C', 'category': 'balanced'},
                    {'cid': 753, 'name': 'Polystyrene', 'molecular_weight': 104000, 'smiles': 'C=Cc1ccccc1', 'category': 'balanced'},
                    {'cid': 31376, 'name': 'Polyvinyl Chloride', 'molecular_weight': 62000, 'smiles': 'C=CCl', 'category': 'balanced'},
                    {'cid': 62787, 'name': 'Nylon-6', 'molecular_weight': 22600, 'smiles': 'C1CCC(=O)N1', 'category': 'specialist'},
                    {'cid': 25312, 'name': 'Polycarbonate', 'molecular_weight': 25400, 'smiles': 'CC(C)(C)c1ccc(cc1)C(C)(C)C', 'category': 'specialist'}
                ]
            }
            
            compounds = fallback_compounds.get(material_type.lower(), [])
            
            # Add some general compounds if material type not found
            if not compounds:
                compounds = [
                    {'cid': 887, 'name': 'Methanol', 'molecular_weight': 32.04, 'smiles': 'CO', 'category': 'balanced'},
                    {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                    {'cid': 6344, 'name': 'Ethanol', 'molecular_weight': 46.07, 'smiles': 'CCO', 'category': 'balanced'}
                ]
            
            return compounds
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced fallback compounds: {e}")
            return []
