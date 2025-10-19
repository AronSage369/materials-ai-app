import google.generativeai as genai
import json
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np

class AdvancedStrategist:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.set_api_key(api_key)
        
    def set_api_key(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    def deep_scientific_analysis(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Deep scientific analysis with multi-perspective thinking"""
        # Create the JSON example as a separate string to avoid formatting issues
        json_example = '''{
            "quantum_analysis": "Detailed quantum chemistry requirements",
            "molecular_design": "Specific molecular design principles",
            "synthetic_strategy": "Synthetic chemistry approach",
            "material_architecture": "Material structure and morphology",
            "innovative_concepts": ["novel concept 1", "novel concept 2"],
            "critical_functional_groups": ["group1", "group2", "group3"],
            "key_molecular_classes": ["class1", "class2", "class3"],
            "advanced_search_terms": {
                "quantum_terms": ["term1", "term2"],
                "structural_terms": ["term3", "term4"],
                "functional_terms": ["term5", "term6"],
                "innovative_terms": ["unconventional1", "unconventional2"]
            },
            "property_requirements": {
                "electronic_properties": ["property1", "property2"],
                "structural_properties": ["property3", "property4"],
                "stability_properties": ["property5", "property6"]
            },
            "challenge_breakdown": {
                "primary_challenge": "main scientific challenge",
                "secondary_challenges": ["challenge1", "challenge2"],
                "solution_strategies": ["strategy1", "strategy2"]
            }
        }'''
        
        prompt = f"""
        You are a team of expert materials scientists, computational chemists, and AI researchers.
        Analyze this challenge from multiple scientific perspectives:

        CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}

        Think through these perspectives:

        1. QUANTUM CHEMISTRY PERSPECTIVE:
        - What electronic structures are needed?
        - What molecular orbitals and band structures?
        - What charge transport mechanisms?

        2. MOLECULAR ENGINEERING PERSPECTIVE:
        - What functional groups are essential?
        - What molecular architectures?
        - What supramolecular interactions?

        3. SYNTHETIC CHEMISTRY PERSPECTIVE:
        - What synthetic pathways are feasible?
        - What precursors and reactions?
        - What purification methods?

        4. APPLIED MATERIALS PERSPECTIVE:
        - What processing conditions?
        - What scalability considerations?
        - What stability requirements?

        5. INNOVATIVE THINKING PERSPECTIVE:
        - What unconventional approaches?
        - What bio-inspired designs?
        - What multi-functional systems?

        Return a comprehensive JSON analysis with the following structure:

        {json_example}

        Be exceptionally detailed, scientific, and innovative!
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                self.logger.warning("Could not extract JSON from response, using fallback")
                return self._create_comprehensive_fallback(challenge_text, material_type)
                
        except Exception as e:
            self.logger.error(f"Deep analysis failed: {e}")
            return self._create_comprehensive_fallback(challenge_text, material_type)

    def generate_innovative_search_strategy(self, scientific_analysis: Dict) -> Dict[str, Any]:
        """Generate innovative search strategy based on scientific analysis"""
        # Create JSON example separately to avoid formatting issues
        json_example = '''{
            "search_dimensions": {
                "conventional": ["term1", "term2"],
                "emerging": ["term3", "term4"],
                "bio_inspired": ["term5", "term6"],
                "multi_functional": ["term7", "term8"],
                "unconventional": ["term9", "term10"]
            },
            "compound_class_priorities": [
                {"class": "class1", "priority": "high", "reasoning": "reasoning1"},
                {"class": "class2", "priority": "medium", "reasoning": "reasoning2"}
            ],
            "molecular_complexity_targets": {
                "simple_molecules": "when to use",
                "complex_architectures": "when to use",
                "hybrid_systems": "when to use"
            },
            "innovation_focus_areas": ["area1", "area2", "area3"]
        }'''
        
        prompt = f"""
        Based on this scientific analysis, create an innovative search strategy:

        {json.dumps(scientific_analysis, indent=2)}

        Generate search strategies that explore:

        1. CONVENTIONAL SOLUTIONS: Established compound classes
        2. EMERGING MATERIALS: Recent discoveries and novel compounds
        3. BIO-INSPIRED DESIGNS: Nature-inspired molecules
        4. MULTI-FUNCTIONAL SYSTEMS: Compounds with multiple functionalities
        5. UNCONVENTIONAL APPROACHES: Surprising and innovative combinations

        Return JSON with the following structure:

        {json_example}
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                self.logger.warning("Could not extract JSON from search strategy response, using fallback")
                return self._create_search_fallback(scientific_analysis)
                
        except Exception as e:
            self.logger.error(f"Search strategy generation failed: {e}")
            return self._create_search_fallback(scientific_analysis)

    def _create_comprehensive_fallback(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Comprehensive fallback analysis"""
        return {
            "quantum_analysis": f"Quantum requirements for {material_type} in {challenge_text}",
            "molecular_design": f"Molecular design for {material_type}",
            "synthetic_strategy": "Standard synthetic approaches",
            "material_architecture": f"Architecture for {material_type}",
            "innovative_concepts": [f"Innovative {material_type} design"],
            "critical_functional_groups": ["functional groups based on application"],
            "key_molecular_classes": [material_type, "advanced materials"],
            "advanced_search_terms": {
                "quantum_terms": [material_type, "electronic", "conductor"],
                "structural_terms": ["polymer", "composite", "nanomaterial"],
                "functional_terms": ["functional", "active", "responsive"],
                "innovative_terms": ["novel", "advanced", "emerging"]
            },
            "property_requirements": {
                "electronic_properties": ["conductivity", "band_gap"],
                "structural_properties": ["stability", "morphology"],
                "stability_properties": ["thermal", "chemical"]
            },
            "challenge_breakdown": {
                "primary_challenge": f"Developing effective {material_type}",
                "secondary_challenges": ["synthesis", "stability", "performance"],
                "solution_strategies": ["molecular design", "composite approach", "nanostructuring"]
            }
        }

    def _create_search_fallback(self, scientific_analysis: Dict) -> Dict[str, Any]:
        """Fallback search strategy"""
        material_type = "material"
        if "key_molecular_classes" in scientific_analysis and scientific_analysis["key_molecular_classes"]:
            material_type = scientific_analysis["key_molecular_classes"][0]
        
        return {
            "search_dimensions": {
                "conventional": [material_type],
                "emerging": ["nanomaterial", "2D material", "quantum material"],
                "bio_inspired": ["bio-inspired", "biomimetic", "natural compound"],
                "multi_functional": ["multifunctional", "smart material", "responsive"],
                "unconventional": ["unconventional", "novel", "breakthrough"]
            },
            "compound_class_priorities": [
                {"class": material_type, "priority": "high", "reasoning": "Primary material type"}
            ],
            "molecular_complexity_targets": {
                "simple_molecules": "For basic applications and quick testing",
                "complex_architectures": "For advanced functionality and specific properties",
                "hybrid_systems": "For multi-functional applications"
            },
            "innovation_focus_areas": ["molecular design", "composite systems", "advanced functionality"]
        }
