# compatibility_checker.py - Chemical compatibility validation
import re
from typing import Dict, List, Any
import streamlit as st

class CompatibilityChecker:
    def __init__(self):
        self.incompatibility_rules = self._load_incompatibility_rules()
        self.functional_groups = self._load_functional_groups()
    
    def validate_all_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Validate compatibility for all formulations"""
        for formulation in formulations:
            compounds = formulation.get('compounds', [])
            ratios = formulation.get('ratios', [1.0] * len(compounds))
            
            if len(compounds) > 1:  # Only check mixtures
                feasibility = self.validate_mixture(compounds, ratios)
                formulation['feasibility'] = feasibility
            else:
                # Single compound - generally compatible
                formulation['feasibility'] = {
                    'homogeneous': True,
                    'chemically_stable': True,
                    'phase_stable': True,
                    'compatibility_issues': [],
                    'risk_score': 0.1
                }
        
        return formulations
    
    def validate_mixture(self, compounds: List, ratios: List[float]) -> Dict[str, Any]:
        """Validate chemical compatibility of mixture"""
        
        issues = []
        risk_factors = []
        
        # Check pairwise compatibility
        for i in range(len(compounds)):
            for j in range(i + 1, len(compounds)):
                comp1, comp2 = compounds[i], compounds[j]
                ratio1, ratio2 = ratios[i], ratios[j]
                
                # Check for incompatibilities
                pair_issues = self._check_pair_compatibility(comp1, comp2, ratio1, ratio2)
                issues.extend(pair_issues)
                
                # Calculate risk factors
                risk_level = self._assess_interaction_risk(comp1, comp2)
                if risk_level > 0.5:
                    risk_factors.append(f"High interaction risk between {self._get_compound_name(comp1)} and {self._get_compound_name(comp2)}")
        
        # Check homogeneity
        is_homogeneous = self._check_homogeneity(compounds, ratios)
        if not is_homogeneous:
            issues.append("Potential phase separation or immiscibility")
        
        # Overall risk score
        risk_score = self._calculate_overall_risk(issues, risk_factors, len(compounds))
        
        return {
            'homogeneous': is_homogeneous,
            'chemically_stable': len(issues) == 0,
            'phase_stable': is_homogeneous,
            'compatibility_issues': issues,
            'risk_factors': risk_factors,
            'risk_score': risk_score
        }
    
    def _check_pair_compatibility(self, comp1, comp2, ratio1: float, ratio2: float) -> List[str]:
        """Check compatibility between two compounds"""
        issues = []
        
        # Get compound information
        name1 = self._get_compound_name(comp1).lower()
        name2 = self._get_compound_name(comp2).lower()
        formula1 = getattr(comp1, 'molecular_formula', '')
        formula2 = getattr(comp2, 'molecular_formula', '')
        
        # Check functional group incompatibilities
        fg1 = self._extract_functional_groups(name1, formula1)
        fg2 = self._extract_functional_groups(name2, formula2)
        
        for group1 in fg1:
            for group2 in fg2:
                incompatibility = self._check_functional_group_incompatibility(group1, group2)
                if incompatibility:
                    issues.append(f"Incompatibility: {group1} with {group2} - {incompatibility}")
        
        # Check specific compound incompatibilities
        specific_issues = self._check_specific_incompatibilities(name1, name2)
        issues.extend(specific_issues)
        
        # Check polarity mismatch
        polarity_mismatch = self._check_polarity_mismatch(fg1, fg2, ratio1, ratio2)
        if polarity_mismatch:
            issues.append(polarity_mismatch)
        
        return issues
    
    def _extract_functional_groups(self, name: str, formula: str) -> List[str]:
        """Extract functional groups from compound name and formula"""
        groups = []
        
        # Common functional groups patterns
        group_patterns = {
            'acid': r'(acid|carboxylic)',
            'alcohol': r'(alcohol|hydroxy|ol$)',
            'aldehyde': r'aldehyde',
            'ketone': r'ketone',
            'ester': r'ester',
            'ether': r'ether',
            'amine': r'amine',
            'amide': r'amide',
            'siloxane': r'siloxane|silicone',
            'halogenated': r'fluor|chlor|brom|iod',
            'aromatic': r'benz|phenyl|aryl',
            'saturated': r'alkane|paraffin',
            'unsaturated': r'ene|yne|unsaturat'
        }
        
        for group, pattern in group_patterns.items():
            if re.search(pattern, name, re.IGNORECASE):
                groups.append(group)
        
        # Additional detection from formula
        if 'O' in formula and 'C' in formula:
            if formula.count('O') >= 2 and 'acid' not in groups:
                groups.append('oxygenated')
        
        if 'N' in formula:
            groups.append('nitrogenated')
        
        if 'S' in formula:
            groups.append('sulfur')
        
        if 'F' in formula:
            groups.append('fluorinated')
        
        if 'Cl' in formula or 'Br' in formula or 'I' in formula:
            groups.append('halogenated')
        
        return groups
    
    def _check_functional_group_incompatibility(self, group1: str, group2: str) -> str:
        """Check if two functional groups are incompatible"""
        incompatibilities = {
            'acid': ['amine', 'alcohol'],  # Acids react with bases and alcohols
            'amine': ['acid', 'aldehyde'],  # Amines react with acids and aldehydes
            'alcohol': ['acid'],  # Alcohols can esterify with acids
            'aldehyde': ['amine'],  # Aldehydes react with amines
            'halogenated': ['amine', 'alcohol'],  # Halogenated compounds can react
        }
        
        if group1 in incompatibilities and group2 in incompatibilities[group1]:
            return f"Possible reaction between {group1} and {group2}"
        if group2 in incompatibilities and group1 in incompatibilities[group2]:
            return f"Possible reaction between {group1} and {group2}"
        
        return ""
    
    def _check_specific_incompatibilities(self, name1: str, name2: str) -> List[str]:
        """Check for specific known incompatibilities"""
        issues = []
        
        # Strong acid-base combinations
        acid_terms = ['sulfuric', 'hydrochloric', 'nitric', 'phosphoric', 'acid']
        base_terms = ['sodium hydroxide', 'potassium hydroxide', 'amine', 'ammonia']
        
        if any(term in name1 for term in acid_terms) and any(term in name2 for term in base_terms):
            issues.append("Strong acid-base combination - neutralization reaction likely")
        
        # Oxidizer-reducer combinations
        oxidizer_terms = ['peroxide', 'nitrate', 'chlorate', 'permanganate']
        reducer_terms = ['alcohol', 'aldehyde', 'sulfide', 'amine']
        
        if any(term in name1 for term in oxidizer_terms) and any(term in name2 for term in reducer_terms):
            issues.append("Oxidizer-reducer combination - potential redox reaction")
        
        # Water-sensitive compounds
        water_sensitive = ['anhydride', 'acid chloride', 'alkali metal']
        aqueous = ['water', 'aqueous', 'solution']
        
        if any(term in name1 for term in water_sensitive) and any(term in name2 for term in aqueous):
            issues.append("Water-sensitive compound in aqueous environment")
        
        return issues
    
    def _check_polarity_mismatch(self, fg1: List[str], fg2: List[str], ratio1: float, ratio2: float) -> str:
        """Check for polarity mismatches that could cause phase separation"""
        polar_groups = ['acid', 'alcohol', 'amine', 'amide', 'aldehyde', 'ketone']
        nonpolar_groups = ['saturated', 'siloxane', 'aromatic']
        
        has_polar = any(group in polar_groups for group in fg1 + fg2)
        has_nonpolar = any(group in nonpolar_groups for group in fg1 + fg2)
        
        if has_polar and has_nonpolar:
            # Check if one component dominates
            if ratio1 > 0.7 or ratio2 > 0.7:
                return "Polar-nonpolar mixture with one component dominant - likely miscible"
            else:
                return "Polar-nonpolar mixture - potential miscibility issues"
        
        return ""
    
    def _check_homogeneity(self, compounds: List, ratios: List[float]) -> bool:
        """Check if mixture is likely homogeneous"""
        if len(compounds) == 1:
            return True
        
        # Simple heuristic based on compound types
        polar_count = 0
        nonpolar_count = 0
        
        for compound in compounds:
            name = self._get_compound_name(compound).lower()
            formula = getattr(compound, 'molecular_formula', '')
            fg = self._extract_functional_groups(name, formula)
            
            if any(group in ['acid', 'alcohol', 'amine', 'amide'] for group in fg):
                polar_count += 1
            elif any(group in ['saturated', 'siloxane'] for group in fg):
                nonpolar_count += 1
        
        # Mixed polar/nonpolar systems might separate
        if polar_count > 0 and nonpolar_count > 0:
            return False
        
        return True
    
    def _assess_interaction_risk(self, comp1, comp2) -> float:
        """Assess risk level of chemical interaction"""
        risk = 0.0
        
        name1 = self._get_compound_name(comp1).lower()
        name2 = self._get_compound_name(comp2).lower()
        
        # High risk combinations
        high_risk_terms = [
            ('acid', 'base'), ('oxidizer', 'reducer'), 
            ('peroxide', 'organic'), ('chlorate', 'organic')
        ]
        
        for term1, term2 in high_risk_terms:
            if (term1 in name1 and term2 in name2) or (term1 in name2 and term2 in name1):
                risk += 0.8
        
        # Medium risk - reactive functional groups
        fg1 = self._extract_functional_groups(name1, getattr(comp1, 'molecular_formula', ''))
        fg2 = self._extract_functional_groups(name2, getattr(comp2, 'molecular_formula', ''))
        
        reactive_pairs = [('acid', 'amine'), ('aldehyde', 'amine'), ('alcohol', 'acid')]
        for pair in reactive_pairs:
            if pair[0] in fg1 and pair[1] in fg2:
                risk += 0.4
            if pair[1] in fg1 and pair[0] in fg2:
                risk += 0.4
        
        return min(1.0, risk)
    
    def _calculate_overall_risk(self, issues: List[str], risk_factors: List[str], num_compounds: int) -> float:
        """Calculate overall risk score"""
        base_risk = 0.1  # Base risk for any mixture
        
        # Add risk from issues
        issue_risk = len(issues) * 0.1
        
        # Add risk from factors
        factor_risk = len(risk_factors) * 0.15
        
        # Complexity risk - more compounds = higher risk
        complexity_risk = (num_compounds - 1) * 0.05
        
        total_risk = base_risk + issue_risk + factor_risk + complexity_risk
        
        return min(1.0, total_risk)
    
    def _get_compound_name(self, compound) -> str:
        """Get compound name for display"""
        if hasattr(compound, 'iupac_name') and compound.iupac_name:
            return compound.iupac_name
        elif hasattr(compound, 'synonyms') and compound.synonyms:
            return compound.synonyms[0]
        else:
            return f"Compound_{getattr(compound, 'cid', 'unknown')}"
    
    def _load_incompatibility_rules(self) -> Dict:
        """Load chemical incompatibility rules"""
        return {
            'acids': ['bases', 'cyanides', 'sulfides'],
            'bases': ['acids', 'ammonium_salts'],
            'oxidizers': ['organics', 'reducers', 'combustibles'],
            'reducers': ['oxidizers', 'nitrates', 'chlorates']
        }
    
    def _load_functional_groups(self) -> Dict:
        """Load functional group database"""
        return {
            'acid': {'reactivity': 'high', 'polarity': 'high'},
            'alcohol': {'reactivity': 'medium', 'polarity': 'high'},
            'aldehyde': {'reactivity': 'high', 'polarity': 'medium'},
            'ketone': {'reactivity': 'low', 'polarity': 'medium'},
            'ester': {'reactivity': 'low', 'polarity': 'medium'},
            'ether': {'reactivity': 'low', 'polarity': 'low'},
            'amine': {'reactivity': 'high', 'polarity': 'high'},
            'amide': {'reactivity': 'low', 'polarity': 'high'},
            'siloxane': {'reactivity': 'low', 'polarity': 'low'}
        }
