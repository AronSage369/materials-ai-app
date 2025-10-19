import google.generativeai as genai
import json
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from utils import cached

class EnhancedAIStrategist:
    """
    Advanced AI strategist that uses multi-step reasoning and creative thinking
    to develop innovative materials solutions
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.set_api_key(api_key)
        
    def set_api_key(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise

    @cached
    def deep_scientific_analysis(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """
        Perform deep scientific analysis using multi-step reasoning
        """
        prompt = f"""
        You are an EXPERT computational materials scientist, chemist, and AI research pioneer.
        Your task is to perform DEEP SCIENTIFIC ANALYSIS of this materials challenge.

        CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}

        Think through this problem step by step with INNOVATIVE and UNCONVENTIONAL approaches:

        STEP 1: DEEP SCIENTIFIC UNDERSTANDING
        - What are the FUNDAMENTAL physical/chemical principles at play?
        - What molecular mechanisms and quantum effects are relevant?
        - What are the key structure-property relationships?
        - Consider electronic structure, band theory, molecular orbitals, interfacial effects

        STEP 2: BREAKTHROUGH THINKING
        - What UNCONVENTIONAL material classes could work?
        - What BIO-INSPIRED or NATURE-MIMICKING approaches could help?
        - What NANOSCALE or QUANTUM effects could be leveraged?
        - What MULTIFUNCTIONAL or SMART material concepts apply?

        STEP 3: ADVANCED SEARCH STRATEGY
        - Generate DIVERSE and INNOVATIVE search terms covering:
          * Electronic/optical properties
          * Structural motifs
          * Functional groups
          * Application-specific terms
          * Emerging material classes
        - Think BEYOND conventional classifications

        STEP 4: CREATIVE FORMULATION APPROACH
        - What NOVEL combinations could create synergistic effects?
        - How can we design MULTI-COMPONENT systems?
        - What SOLUTE-SOLVENT-ADDITIVE combinations?
        - Consider COMPOSITES, HYBRIDS, NANOCOMPOSITES

        Return a RICH, DETAILED JSON with this structure:
        {{
            "deep_scientific_analysis": "Comprehensive scientific reasoning with quantum, molecular, and structural insights",
            "key_physical_mechanisms": ["list", "of", "fundamental", "mechanisms"],
            "innovative_approaches": ["unconventional_approach1", "unconventional_approach2"],
            "target_material_classes": ["advanced_class1", "emerging_class2", "unconventional_class3"],
            "specific_functional_groups": ["group1", "group2", "group3"],
            "quantum_considerations": ["quantum_effect1", "quantum_effect2"],
            "search_strategy": {{
                "electronic_properties": ["term1", "term2", "term3"],
                "structural_features": ["feature1", "feature2", "feature3"],
                "functional_groups": ["group1", "group2", "group3"],
                "application_terms": ["application1", "application2"],
                "innovative_concepts": ["concept1", "concept2", "concept3"]
            }},
            "formulation_philosophy": "Description of innovative formulation strategy",
            "synergistic_combinations": ["combination1", "combination2"],
            "predicted_challenges": ["challenge1", "challenge2"],
            "breakthrough_ideas": ["innovative_idea1", "innovative_idea2", "innovative_idea3"]
        }}

        Be CREATIVE, SCIENTIFICALLY RIGOROUS, and INNOVATIVE!
        Think like a Nobel Prize winner discovering new material paradigms!
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                strategy = json.loads(json_str)
                return strategy
            else:
                return self._create_innovative_fallback(challenge_text, material_type)
                
        except Exception as e:
            self.logger.error(f"Error in deep scientific analysis: {e}")
            return self._create_innovative_fallback(challenge_text, material_type)

    @cached
    def generate_quantum_insights(self, formulation: Dict, challenge: str) -> Dict[str, Any]:
        """
        Generate quantum mechanical insights for formulations
        """
        composition = formulation.get('composition', [])
        comp_names = [comp.get('name', 'Unknown') for comp in composition]
        
        prompt = f"""
        As a QUANTUM CHEMISTRY expert, analyze this formulation for electronic and quantum properties:

        FORMULATION: {', '.join(comp_names)}
        CHALLENGE: {challenge}

        Analyze from FIRST PRINCIPLES:

        1. ELECTRONIC STRUCTURE:
        - Predicted HOMO-LUMO gaps and band structures
        - Charge transport mechanisms
        - Exciton formation and dynamics
        - Interface electronic properties

        2. QUANTUM EFFECTS:
        - Quantum confinement possibilities
        - Spin-related phenomena
        - Coherence and entanglement considerations
        - Many-body effects

        3. MOLECULAR INTERACTIONS:
        - Orbital overlap and hybridization
        - Charge transfer complexes
        - Polaronic effects
        - Defect states and trapping

        Return detailed quantum insights:
        {{
            "electronic_structure": {{
                "predicted_band_gap": "estimated range with reasoning",
                "charge_transport": "mechanisms and efficiency",
                "interface_properties": "electronic interface behavior"
            }},
            "quantum_phenomena": ["phenomenon1", "phenomenon2"],
            "molecular_orbital_analysis": "HOMO-LUMO and orbital interactions",
            "quantum_efficiency_factors": ["factor1", "factor2"],
            "recommendations_for_enhancement": ["recommendation1", "recommendation2"]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error generating quantum insights: {e}")
            return {}

    @cached
    def creative_material_design(self, base_strategy: Dict, compounds_found: List[Dict]) -> Dict[str, Any]:
        """
        Use AI to creatively design material combinations based on available compounds
        """
        compound_names = [comp.get('name', 'Unknown') for comp in compounds_found[:20]]  # Limit to top 20
        
        prompt = f"""
        You are a CREATIVE MATERIALS DESIGNER AI. Design innovative formulations using available compounds.

        BASE STRATEGY: {json.dumps(base_strategy, indent=2)}
        AVAILABLE COMPOUNDS: {', '.join(compound_names)}

        Design INNOVATIVE formulations by thinking CREATIVELY:

        1. UNCONVENTIONAL COMBINATIONS:
        - Mix different chemical classes for emergent properties
        - Create multi-phase systems
        - Design core-shell or layered structures

        2. SYNERGISTIC EFFECTS:
        - Identify compounds that could work together synergistically
        - Design energy/electron transfer systems
        - Create catalytic or amplifying combinations

        3. NANOSCALE ENGINEERING:
        - Design nanocomposite approaches
        - Consider quantum dot or nanoparticle incorporation
        - Plan interfacial engineering

        4. MULTIFUNCTIONAL DESIGN:
        - Combine electronic, optical, and mechanical functions
        - Design stimuli-responsive systems
        - Create self-healing or adaptive materials

        Return creative formulation strategies:
        {{
            "innovative_combinations": [
                {{
                    "name": "creative combination name",
                    "components": ["compound1", "compound2", "compound3"],
                    "rationale": "scientific reasoning for this combination",
                    "expected_properties": ["property1", "property2"],
                    "synergistic_mechanisms": ["mechanism1", "mechanism2"]
                }}
            ],
            "nanoscale_designs": ["design1", "design2"],
            "multifunctional_approaches": ["approach1", "approach2"],
            "emerging_property_predictions": ["prediction1", "prediction2"]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error in creative material design: {e}")
            return {}

    def _create_innovative_fallback(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Create innovative fallback strategy"""
        return {
            "deep_scientific_analysis": f"Advanced analysis for {material_type} addressing {challenge_text}",
            "key_physical_mechanisms": ["electronic transport", "optical absorption", "quantum effects"],
            "innovative_approaches": ["nanocomposite design", "multifunctional materials", "bio-inspired structures"],
            "target_material_classes": ["advanced polymers", "quantum materials", "hybrid composites"],
            "search_strategy": {
                "electronic_properties": ["semiconductor", "conductor", "photovoltaic"],
                "structural_features": ["nanostructured", "porous", "layered"],
                "functional_groups": ["conjugated", "aromatic", "charge_transfer"],
                "application_terms": [material_type, "advanced material", "functional material"],
                "innovative_concepts": ["emerging materials", "quantum materials", "smart materials"]
            },
            "breakthrough_ideas": [
                "Explore quantum confinement effects",
                "Design multi-component synergistic systems",
                "Incorporate bio-inspired functionality"
            ]
        }
