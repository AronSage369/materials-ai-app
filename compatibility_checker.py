import re
from typing import Dict, List, Any, Tuple
import numpy as np

class CompatibilityChecker:
    def __init__(self):
        # Enhanced incompatibility database
        self.incompatible_pairs = {
            # Acid-Base reactions
            'acid_base': {
                'groups1': ['carboxylic_acid', 'sulfonic_acid', 'phosphonic_acid', 'strong_acid'],
                'groups2': ['amine', 'hydroxide', 'strong_base', 'weak_base'],
                'risk': 0.8,
                'description': 'Acid-base reaction may cause neutralization, heat generation, or salt formation'
            },
            # Redox reactions
            'redox': {
                'groups1': ['strong_oxidizer', 'peroxide', 'nitro', 'halogen'],
                'groups2': ['reducing_agent', 'alcohol', 'aldehyde', 'thiol'],
                'risk': 0.9,
                'description': 'Redox reaction may cause fire, explosion, or gas generation'
            },
            # Water-sensitive compounds
            'hydrolysis': {
                'groups1': ['acid_chloride', 'anhydride', 'alkali_metal', 'metal_hydride'],
                'groups2': ['water', 'moisture'],
                'risk': 0.7,
                'description': 'Hydrolysis may generate heat, acids, or flammable gases'
            },
            # Polymerization initiators
            'polymerization': {
                'groups1': ['peroxide', 'azo', 'radical_initiator'],
                'groups2': ['monomer', 'vinyl', 'acrylate', 'styrene'],
                'risk': 0.8,
                'description': 'May initiate uncontrolled polymerization'
            },
            # Complex formation
            'complexation': {
                'groups1': ['metal_ion', 'transition_metal'],
                'groups2': ['amine', 'carbonyl', 'cyano', 'thiol'],
                'risk': 0.5,
                'description': 'May form complexes affecting properties'
            }
        }
        
        # Functional group patterns
        self.functional_groups = {
            'carboxylic_acid': r'C\(=O\)OH|COOH',
            'amine': r'NH2|NHR|NR2|amine',
            'alcohol': r'OH|hydroxyl|alcohol',
            'aldehyde': r'CH=O|aldehyde',
            'ketone': r'C\(=O\)C|ketone',
            'ester': r'COOC|ester',
            'ether': r'COC|ether',
            'halogen': r'F|Cl|Br|I|halogen',
            'nitro': r'NO2|nitro',
            'sulfonic_acid': r'SO3H|sulfonic',
            'peroxide': r'OO|peroxide',
            'water': r'H2O|water',
            'strong_acid': r'sulfuric acid|hydrochloric acid|nitric acid',
            'strong_base': r'sodium hydroxide|potassium hydroxide|NaOH|KOH'
        }

    def validate_all_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Validate chemical compatibility for all formulations"""
        for formulation in formulations:
            if len(formulation.get('composition', [])) > 1:
                validation_result = self.validate_mixture(formulation)
                formulation.update(validation_result)
            else:
                # Single component - generally compatible
                formulation.update({
                    'compatibility_risk': 0.1,
                    'compatibility_issues': [],
                    'compatibility_warnings': ['Single component - generally stable']
                })
                
        return formulations

    def validate_mixture(self, formulation: Dict) -> Dict[str, Any]:
        """Validate chemical compatibility of a mixture"""
        composition = formulation.get('composition', [])
        issues = []
        total_risk = 0.0
        checked_pairs = set()
        
        # Check all pairwise combinations
        for i in range(len(composition)):
            for j in range(i + 1, len(composition)):
                comp1, comp2 = composition[i], composition[j]
                pair_key = tuple(sorted([comp1.get('cid'), comp2.get('cid')]))
                
                if pair_key in checked_pairs:
                    continue
                    
                checked_pairs.add(pair_key)
                
                # Check compatibility for this pair
                pair_issues, pair_risk = self._check_pair_compatibility(comp1, comp2)
                issues.extend(pair_issues)
                total_risk = max(total_risk, pair_risk)  # Use maximum risk from any pair
        
        # Calculate overall risk considering mixture complexity
        complexity_factor = self._calculate_complexity_factor(composition)
        overall_risk = min(1.0, total_risk * complexity_factor)
        
        # Generate warnings based on risk level
        warnings = self._generate_compatibility_warnings(overall_risk, issues)
        
        return {
            'compatibility_risk': overall_risk,
            'compatibility_issues': issues,
            'compatibility_warnings': warnings,
            'compatibility_summary': self._generate_compatibility_summary(overall_risk)
        }

    def _check_pair_compatibility(self, comp1: Dict, comp2: Dict) -> Tuple[List[str], float]:
        """Check compatibility between two compounds"""
        issues = []
        max_risk = 0.0
        
        # Extract functional groups from names and SMILES
        groups1 = self._extract_functional_groups(comp1)
        groups2 = self._extract_functional_groups(comp2)
        
        # Check against known incompatibility patterns
        for reaction_type, pattern in self.incompatible_pairs.items():
            risk = pattern['risk']
            
            # Check if both compounds have matching functional groups
            has_group1 = any(g in groups1 for g in pattern['groups1'])
            has_group2 = any(g in groups2 for g in pattern['groups2'])
            
            # Check reverse combination
            has_reverse_group1 = any(g in groups1 for g in pattern['groups2'])
            has_reverse_group2 = any(g in groups2 for g in pattern['groups1'])
            
            if (has_group1 and has_group2) or (has_reverse_group1 and has_reverse_group2):
                issue_desc = f"{pattern['description']} between {comp1.get('name')} and {comp2.get('name')}"
                issues.append(issue_desc)
                max_risk = max(max_risk, risk)
        
        # Check for specific dangerous combinations
        specific_issues, specific_risk = self._check_specific_incompatibilities(comp1, comp2)
        issues.extend(specific_issues)
        max_risk = max(max_risk, specific_risk)
        
        return issues, max_risk

    def _extract_functional_groups(self, compound: Dict) -> List[str]:
        """Extract functional groups from compound data"""
        groups = []
        name = compound.get('name', '').lower()
        smiles = compound.get('smiles', '').lower()
        iupac = compound.get('iupac_name', '').lower()
        
        text_to_search = f"{name} {smiles} {iupac}"
        
        for group_name, pattern in self.functional_groups.items():
            if re.search(pattern, text_to_search, re.IGNORECASE):
                groups.append(group_name)
        
        return groups

    def _check_specific_incompatibilities(self, comp1: Dict, comp2: Dict) -> Tuple[List[str], float]:
        """Check for specific known dangerous combinations"""
        issues = []
        risk = 0.0
        
        name1 = comp1.get('name', '').lower()
        name2 = comp2.get('name', '').lower()
        
        # Known dangerous combinations
        dangerous_pairs = [
            (['nitric acid', 'sulfuric acid'], ['organic material', 'alcohol', 'acetone'], 0.95, 'May cause violent oxidation or explosion'),
            (['chlorine', 'hypochlorite'], ['ammonia', 'amine'], 0.9, 'May form explosive chloramines'),
            (['peroxide', 'organic peroxide'], ['metal', 'transition metal'], 0.8, 'May catalyze explosive decomposition'),
            (['strong acid'], ['cyanide'], 0.95, 'May release toxic hydrogen cyanide gas'),
            (['strong base'], ['chlorinated solvent'], 0.7, 'May cause dehydrohalogenation')
        ]
        
        for pair1, pair2, pair_risk, description in dangerous_pairs:
            if (any(p in name1 for p in pair1) and any(p in name2 for p in pair2) or
                any(p in name1 for p in pair2) and any(p in name2 for p in pair1)):
                issues.append(f"{description}: {name1} + {name2}")
                risk = max(risk, pair_risk)
        
        return issues, risk

    def _calculate_complexity_factor(self, composition: List[Dict]) -> float:
        """Calculate risk factor based on mixture complexity"""
        num_components = len(composition)
        
        # More components = higher complexity risk
        if num_components == 1:
            return 1.0
        elif num_components == 2:
            return 1.2
        elif num_components == 3:
            return 1.5
        else:
            return 1.5 + (num_components - 3) * 0.2

    def _generate_compatibility_warnings(self, risk: float, issues: List[str]) -> List[str]:
        """Generate appropriate warnings based on risk level"""
        if risk < 0.3:
            return ["Low compatibility risk - formulation appears stable"]
        elif risk < 0.6:
            warnings = ["Medium compatibility risk - review recommended"]
            if issues:
                warnings.append("Consider experimental validation")
            return warnings
        else:
            warnings = ["HIGH COMPATIBILITY RISK - EXTREME CAUTION REQUIRED"]
            if issues:
                warnings.extend([f"⚠️ {issue}" for issue in issues[:3]])  # Show top 3 issues
            warnings.append("Experimental validation MANDATORY before use")
            return warnings

    def _generate_compatibility_summary(self, risk: float) -> str:
        """Generate a compatibility summary"""
        if risk < 0.3:
            return "Compatible"
        elif risk < 0.6:
            return "Moderate Risk"
        elif risk < 0.8:
            return "High Risk"
        else:
            return "Extreme Risk"
