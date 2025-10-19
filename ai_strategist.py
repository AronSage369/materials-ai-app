import google.generativeai as genai
import json
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np

class AIStrategist:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.set_api_key(api_key)
        
    def set_api_key(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    def think_about_challenge(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Advanced AI thinking with innovative approaches"""
        prompt = f"""
        You are an EXPERT computational materials scientist with NO LIMITATIONS on creativity.
        Think BEYOND conventional solutions for this challenge:

        CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}

        Think INNOVATIVELY:

        1. BREAKTHROUGH THINKING:
        - What would a Nobel Prize-winning solution look like?
        - What unconventional material combinations?
        - What emerging technologies could be leveraged?

        2. MULTI-SCALE DESIGN:
        - Atomic-level design principles
        - Molecular assembly strategies
        - Macroscopic material architecture

        3. CROSS-DOMAIN INNOVATION:
        - Borrow concepts from biology, physics, electronics
        - Combine organic/inorganic/hybrid systems
        - Leverage quantum effects and nanoscale phenomena

        4. ADAPTIVE STRATEGY:
        - How to explore the ENTIRE chemical space?
        - What smart search strategies?
        - How to balance exploration vs exploitation?

        Return a POWERFUL strategy:

        {{
            "breakthrough_vision": "Visionary approach description",
            "unconventional_approaches": ["approach1", "approach2", "approach3"],
            "cross_domain_ideas": ["biology-inspired", "quantum-enabled", "nanoscale-designed"],
            "chemical_space_exploration": {{
                "diversity_strategy": "How to ensure chemical diversity",
                "innovation_focus": "Areas for breakthrough innovation",
                "risk_taking": "How much unconventional risk to take"
            }},
            "smart_search_strategy": {{
                "primary_focus": ["high-priority terms"],
                "secondary_exploration": ["broad exploration terms"],
                "innovative_gambles": ["high-risk high-reward terms"]
            }},
            "formulation_philosophy": {{
                "mixing_strategy": "How to combine components innovatively",
                "complexity_approach": "Simple vs complex formulations",
                "synergy_focus": "What synergistic effects to target"
            }},
            "success_metrics": {{
                "performance_targets": ["target1", "target2"],
                "innovation_metrics": ["novelty", "elegance", "effectiveness"],
                "feasibility_considerations": ["synthesis", "stability", "scalability"]
            }}
        }}

        Be BOLD, CREATIVE, and SCIENTIFICALLY RIGOROUS!
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group())
                # Enhance with specific search terms
                strategy['dynamic_search_terms'] = self._generate_dynamic_search_terms(strategy, challenge_text)
                return strategy
            else:
                return self._create_innovative_fallback(challenge_text, material_type)
                
        except Exception as e:
            self.logger.error(f"Innovative thinking failed: {e}")
            return self._create_innovative_fallback(challenge_text, material_type)

    def _generate_dynamic_search_terms(self, strategy: Dict, challenge_text: str) -> Dict[str, List[str]]:
        """Generate dynamic search terms based on strategy"""
        # Extract key concepts from the strategy
        unconventional = strategy.get('unconventional_approaches', [])
        cross_domain = strategy.get('cross_domain_ideas', [])
        
        # Generate innovative search terms
        search_terms = {
            "breakthrough_terms": [],
            "synergistic_terms": [],
            "emergent_terms": [],
            "quantum_terms": [],
            "bio_terms": []
        }
        
        # Add terms based on strategy content
        for approach in unconventional:
            words = approach.lower().split()
            search_terms["breakthrough_terms"].extend(words)
            
        for idea in cross_domain:
            if 'bio' in idea.lower():
                search_terms["bio_terms"].extend(["biomimetic", "bioinspired", "enzyme", "protein"])
            if 'quantum' in idea.lower():
                search_terms["quantum_terms"].extend(["quantum", "nanoscale", "2D", "graphene"])
            if 'nano' in idea.lower():
                search_terms["emergent_terms"].extend(["nanoparticle", "nanotube", "nanosheet"])
                
        # Ensure uniqueness
        for key in search_terms:
            search_terms[key] = list(set(search_terms[key]))
            
        return search_terms

    def _create_innovative_fallback(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Fallback that's still innovative"""
        return {
            "breakthrough_vision": f"Create revolutionary {material_type} for {challenge_text}",
            "unconventional_approaches": [
                "hybrid organic-inorganic systems",
                "multi-functional composites",
                "stimuli-responsive materials"
            ],
            "cross_domain_ideas": [
                "biological self-assembly principles",
                "quantum confinement effects",
                "nanoscale interface engineering"
            ],
            "smart_search_strategy": {
                "primary_focus": [material_type, "advanced", "functional"],
                "secondary_exploration": ["nanomaterial", "composite", "hybrid"],
                "innovative_gambles": ["quantum dot", "2D material", "MOF"]
            },
            "dynamic_search_terms": {
                "breakthrough_terms": ["revolutionary", "breakthrough", "novel"],
                "synergistic_terms": ["composite", "hybrid", "blend"],
                "emergent_terms": ["nanomaterial", "quantum", "2D"],
                "quantum_terms": ["electronic", "conductor", "semiconductor"],
                "bio_terms": ["biomimetic", "natural", "enzyme"]
            }
        }

    def evaluate_formulation_science(self, formulation: Dict, strategy: Dict) -> Dict[str, Any]:
        """Advanced scientific evaluation"""
        composition = formulation.get('composition', [])
        comp_names = [comp.get('name', 'Unknown') for comp in composition]
        
        prompt = f"""
        CRITICAL scientific evaluation of this formulation:

        FORMULATION: {', '.join(comp_names)}
        STRATEGY: {strategy.get('breakthrough_vision', 'Unknown')}

        Evaluate with SCIENTIFIC RIGOR:

        1. MOLECULAR COMPATIBILITY:
        - Electronic structure alignment
        - Molecular orbital interactions
        - Supramolecular assembly potential

        2. SYNERGISTIC POTENTIAL:
        - Expected synergistic effects
        - Emergent properties
        - Multi-functional capabilities

        3. QUANTUM CHEMICAL ASSESSMENT:
        - Band structure considerations
        - Charge transport mechanisms
        - Exciton dynamics

        4. PRACTICAL FEASIBILITY:
        - Synthesis pathways
        - Stability under conditions
        - Scalability potential

        Return HONEST assessment:
        {{
            "scientific_merit": "High/Medium/Low with detailed reasoning",
            "innovation_potential": "Breakthrough/Incremental/Minimal",
            "key_synergies": ["synergy1", "synergy2"],
            "quantum_advantages": ["advantage1", "advantage2"],
            "critical_challenges": ["challenge1", "challenge2"],
            "improvement_recommendations": ["recommendation1", "recommendation2"],
            "confidence_in_prediction": 0.0-1.0
        }}

        Be scientifically rigorous and honest!
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "scientific_merit": "Medium - AI evaluation failed",
                    "innovation_potential": "Unknown",
                    "key_synergies": [],
                    "critical_challenges": ["Evaluation system error"],
                    "confidence_in_prediction": 0.3
                }
        except Exception as e:
            self.logger.error(f"Scientific evaluation failed: {e}")
            return {
                "scientific_merit": "Low - Evaluation error",
                "innovation_potential": "Unknown",
                "key_synergies": [],
                "critical_challenges": ["Evaluation system failure"],
                "confidence_in_prediction": 0.1
            }
