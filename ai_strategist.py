import google.generativeai as genai
import json
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from utils import cached, MemoryManager

class AIStrategist:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.set_api_key(api_key)
        
    def set_api_key(self, api_key: str):
        """Configure Gemini API with error handling"""
        try:
            if not api_key:
                raise ValueError("API key cannot be empty")
                
            genai.configure(api_key=api_key)
            # Try to use the model, fallback to simpler model if needed
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                # Test the model with a simple prompt
                test_response = self.model.generate_content("Test")
                self.logger.info("Gemini API configured successfully")
            except Exception as model_error:
                self.logger.warning(f"Flash model failed, trying pro: {model_error}")
                self.model = genai.GenerativeModel('gemini-pro')
                
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    @cached
    def think_about_challenge(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Advanced AI thinking about the challenge - generates search strategies, compound classes, and formulation approaches"""
        prompt = f"""
        You are an expert computational materials scientist and AI research assistant. 
        Analyze this materials challenge and think deeply about the solution.

        CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}

        Think step by step:

        1. UNDERSTAND THE CORE SCIENCE:
        - What are the fundamental physical/chemical principles involved?
        - What molecular mechanisms enable the desired properties?
        - What are the key performance metrics?

        2. IDENTIFY CHEMICAL STRATEGIES:
        - What chemical classes could provide the required functionality?
        - What molecular structures or functional groups are needed?
        - Consider both organic and inorganic options

        3. GENERATE SEARCH STRATEGY:
        - What specific compounds or material classes should we search for?
        - What are the key PubChem search terms?
        - Consider both conventional and unconventional options

        4. FORMULATION APPROACH:
        - Should we use single compounds or mixtures?
        - What combination strategies might work?
        - Consider solute-solvent systems, composites, etc.

        Return a JSON with this structure:
        {{
            "scientific_analysis": "Detailed scientific reasoning",
            "key_mechanisms": ["list", "of", "key", "mechanisms"],
            "target_compound_classes": ["class1", "class2", "class3"],
            "specific_compounds": ["compound1", "compound2"],
            "search_strategy": {{
                "primary_terms": ["term1", "term2"],
                "secondary_terms": ["term3", "term4"],
                "innovative_terms": ["unconventional_term1", "unconventional_term2"]
            }},
            "formulation_approach": "Description of formulation strategy",
            "expected_challenges": ["challenge1", "challenge2"],
            "innovative_ideas": ["idea1", "idea2"]
        }}

        Be specific, scientific, and innovative!
        """

        try:
            if not self.model:
                raise Exception("AI model not initialized")
                
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                strategy = json.loads(json_str)
                return strategy
            else:
                self.logger.warning("No JSON found in AI response, using fallback")
                return self._create_fallback_strategy(challenge_text, material_type)
                
        except Exception as e:
            self.logger.error(f"Error in AI thinking: {e}")
            return self._create_fallback_strategy(challenge_text, material_type)

    @cached
    def predict_property_enhancement(self, formulation: Dict, target_properties: Dict) -> Dict[str, Any]:
        """AI prediction of how formulation modifications might enhance properties"""
        composition = formulation.get('composition', [])
        comp_names = [comp.get('name', 'Unknown') for comp in composition]
        
        prompt = f"""
        As a computational chemist, predict how this formulation might be enhanced:

        CURRENT FORMULATION: {', '.join(comp_names)}
        TARGET PROPERTIES: {json.dumps(target_properties, indent=2)}

        Analyze:
        1. Potential synergistic effects between components
        2. How adding/modifying components could improve properties
        3. What physical/chemical mechanisms could be leveraged
        4. Specific compound suggestions for enhancement

        Return JSON:
        {{
            "synergy_analysis": "Analysis of component interactions",
            "enhancement_strategies": ["strategy1", "strategy2"],
            "suggested_additives": ["additive1", "additive2"],
            "predicted_improvements": {{
                "property1": "expected improvement",
                "property2": "expected improvement"
            }},
            "molecular_mechanisms": ["mechanism1", "mechanism2"]
        }}
        """

        try:
            if not self.model:
                return {}
                
            response = self.model.generate_content(prompt)
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error in property enhancement prediction: {e}")
            return {}

    @cached
    def evaluate_formulation_science(self, formulation: Dict, strategy: Dict) -> Dict[str, Any]:
        """Scientific evaluation of formulation based on computational chemistry principles"""
        composition = formulation.get('composition', [])
        comp_names = [comp.get('name', 'Unknown') for comp in composition]
        
        prompt = f"""
        As a computational materials scientist, provide a scientific evaluation:

        FORMULATION: {', '.join(comp_names)}
        INTENDED APPLICATION: {strategy.get('scientific_analysis', 'Unknown')}

        Evaluate based on:
        1. Molecular compatibility and potential interactions
        2. Electronic structure considerations
        3. Thermodynamic feasibility
        4. Kinetic stability
        5. Potential degradation pathways
        6. Scalability and practical implementation

        Return JSON:
        {{
            "scientific_feasibility": "High/Medium/Low with reasoning",
            "key_advantages": ["advantage1", "advantage2"],
            "potential_issues": ["issue1", "issue2"],
            "molecular_interactions": "Description of molecular-level interactions",
            "stability_assessment": "Stability analysis",
            "improvement_recommendations": ["recommendation1", "recommendation2"]
        }}
        """

        try:
            if not self.model:
                return {}
                
            response = self.model.generate_content(prompt)
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error in scientific evaluation: {e}")
            return {}

    def _create_fallback_strategy(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Fallback strategy when AI thinking fails"""
        self.logger.info("Using fallback strategy")
        return {
            "scientific_analysis": f"Develop {material_type} for {challenge_text}",
            "key_mechanisms": ["basic functionality"],
            "target_compound_classes": [material_type],
            "specific_compounds": [],
            "search_strategy": {
                "primary_terms": [material_type, "chemical compound"],
                "secondary_terms": [],
                "innovative_terms": []
            },
            "formulation_approach": "Standard formulation approach",
            "expected_challenges": ["General implementation challenges"],
            "innovative_ideas": ["Explore standard options"]
        }
