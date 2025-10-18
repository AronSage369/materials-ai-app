import google.generativeai as genai
import json
import re
import random
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class FormulationStrategy:
    primary_objective: str
    key_constraints: List[str]
    search_strategy: str
    target_properties: Dict[str, Dict[str, Any]]
    success_metrics: List[str]
    risk_tolerance: str

class MaterialsAIEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.set_api_key(api_key)
        
        # Agent configurations
        self.agents = {
            'conservative': {
                'name': 'Conservative Agent',
                'description': 'Focuses on safety, stability, and proven combinations',
                'max_components': 3,
                'risk_tolerance': 'low'
            },
            'innovative': {
                'name': 'Innovative Agent', 
                'description': 'Explores novel combinations and unconventional approaches',
                'max_components': 5,
                'risk_tolerance': 'medium'
            },
            'practical': {
                'name': 'Practical Agent',
                'description': 'Balances performance with cost and availability',
                'max_components': 4,
                'risk_tolerance': 'medium'
            },
            'high_performance': {
                'name': 'High-Performance Agent',
                'description': 'Maximizes key performance metrics regardless of cost',
                'max_components': 4,
                'risk_tolerance': 'high'
            },
            'balanced': {
                'name': 'Balanced Agent', 
                'description': 'Seeks optimal trade-offs across all criteria',
                'max_components': 4,
                'risk_tolerance': 'medium'
            }
        }

    def set_api_key(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            # Try multiple model versions for robustness
            model_versions = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
            
            for model_version in model_versions:
                try:
                    self.model = genai.GenerativeModel(model_version)
                    # Test the model with a simple prompt
                    response = self.model.generate_content("Test")
                    if response:
                        print(f"Successfully initialized {model_version}")
                        break
                except Exception as e:
                    print(f"Failed to initialize {model_version}: {e}")
                    continue
                    
            if not self.model:
                raise Exception("Could not initialize any Gemini model")
                
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    def interpret_challenge(self, challenge_text: str, material_type: str, relaxation_level: str) -> Optional[Dict[str, Any]]:
        """Interpret natural language challenge into structured strategy"""
        prompt = f"""
        You are an expert materials scientist AI. Analyze the following material challenge and create a structured strategy.

        MATERIAL CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}
        CONSTRAINT RELAXATION: {relaxation_level}

        Please return a JSON object with the following structure:
        {{
            "primary_objective": "Clear description of the main goal",
            "key_constraints": ["list", "of", "critical", "constraints"],
            "search_strategy": "Detailed approach for compound search",
            "target_properties": {{
                "property_name": {{
                    "target": ideal_value,
                    "min": minimum_acceptable,
                    "max": maximum_acceptable, 
                    "importance": "high/medium/low",
                    "units": "property_units"
                }}
            }},
            "success_metrics": ["list", "of", "success", "criteria"],
            "risk_tolerance": "low/medium/high"
        }}

        Important properties to consider based on material type:
        - Solvents: solubility_parameter, viscosity, boiling_point, flash_point, toxicity
        - Coolants: thermal_conductivity, specific_heat, viscosity, boiling_point, freezing_point
        - Absorbents: absorption_capacity, selectivity, regeneration_temperature, stability
        - Catalysts: activity, selectivity, stability, surface_area, pore_size
        - Polymers: molecular_weight, glass_transition, tensile_strength, thermal_stability

        Ensure all values are numeric where appropriate. Be specific and quantitative.
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
                # Fallback: try to parse the entire response
                try:
                    return json.loads(response_text)
                except:
                    # Final fallback: create basic strategy
                    return self._create_fallback_strategy(challenge_text, material_type)
                    
        except Exception as e:
            print(f"Error in challenge interpretation: {e}")
            return self._create_fallback_strategy(challenge_text, material_type)

    def _create_fallback_strategy(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Create fallback strategy when AI interpretation fails"""
        base_properties = {
            'solvent': {
                'boiling_point': {'min': 50, 'max': 300, 'importance': 'high', 'units': '°C'},
                'viscosity': {'min': 0.1, 'max': 100, 'importance': 'medium', 'units': 'cP'},
                'flash_point': {'min': 40, 'importance': 'high', 'units': '°C'}
            },
            'coolant': {
                'thermal_conductivity': {'min': 0.1, 'importance': 'high', 'units': 'W/m·K'},
                'specific_heat': {'min': 1.0, 'importance': 'high', 'units': 'J/g·K'},
                'viscosity': {'max': 50, 'importance': 'medium', 'units': 'cP'}
            }
        }
        
        return {
            'primary_objective': f"Develop effective {material_type} for: {challenge_text[:100]}...",
            'key_constraints': ['safe handling', 'reasonable cost', 'good stability'],
            'search_strategy': f'Broad search for {material_type} compounds with relevant properties',
            'target_properties': base_properties.get(material_type.lower(), {
                'stability': {'importance': 'high', 'units': 'qualitative'},
                'performance': {'importance': 'high', 'units': 'qualitative'}
            }),
            'success_metrics': ['meets key requirements', 'good safety profile', 'practical formulation'],
            'risk_tolerance': 'medium'
        }

    def multi_agent_formulation_generation(self, compounds: List[Dict], strategy: Dict, innovation_factor: float) -> List[Dict]:
        """Generate formulations using multiple AI agents"""
        all_formulations = []
        
        # Filter compounds by categories
        balanced_compounds = [c for c in compounds if c.get('category') == 'balanced']
        specialist_compounds = [c for c in compounds if c.get('category') == 'specialist']
        
        # Run each agent with weighted probability based on innovation factor
        agent_weights = self._calculate_agent_weights(innovation_factor)
        
        for agent_name, weight in agent_weights.items():
            num_formulations = max(1, int(10 * weight))  # Scale based on weight
            
            for _ in range(num_formulations):
                formulation = getattr(self, f'_{agent_name}_agent')(
                    balanced_compounds, specialist_compounds, strategy
                )
                if formulation:
                    formulation['agent_type'] = self.agents[agent_name]['name']
                    all_formulations.append(formulation)
        
        # Remove duplicates based on composition fingerprint
        unique_formulations = self._remove_duplicate_formulations(all_formulations)
        return unique_formulations

    def _calculate_agent_weights(self, innovation_factor: float) -> Dict[str, float]:
        """Calculate agent weights based on innovation factor"""
        base_weights = {
            'conservative': 0.8 - innovation_factor * 0.6,
            'practical': 1.0,
            'balanced': 1.2,
            'innovative': 0.5 + innovation_factor * 0.8,
            'high_performance': 0.3 + innovation_factor * 0.7
        }
        
        # Normalize weights
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}

    def _conservative_agent(self, balanced_compounds, specialist_compounds, strategy):
        """Conservative agent focusing on safety and simplicity"""
        if not balanced_compounds:
            return None
            
        # Prefer 1-2 component systems
        num_components = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        selected_compounds = random.sample(balanced_compounds, min(num_components, len(balanced_compounds)))
        
        return self._create_formulation_structure(selected_compounds, strategy, 'conservative')

    def _innovative_agent(self, balanced_compounds, specialist_compounds, strategy):
        """Innovative agent exploring novel combinations"""
        all_compounds = balanced_compounds + specialist_compounds
        if not all_compounds:
            return None
            
        # More components and creative ratios
        num_components = random.choices([2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.2])[0]
        selected_compounds = random.sample(all_compounds, min(num_components, len(all_compounds)))
        
        formulation = self._create_formulation_structure(selected_compounds, strategy, 'innovative')
        
        # Add innovative reasoning
        formulation['reasoning'] = "Exploring synergistic effects between multiple components for enhanced performance."
        return formulation

    def _practical_agent(self, balanced_compounds, specialist_compounds, strategy):
        """Practical agent balancing performance and practicality"""
        if not balanced_compounds:
            return None
            
        # Focus on balanced compounds with practical considerations
        num_components = random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        selected_compounds = random.sample(balanced_compounds, min(num_components, len(balanced_compounds)))
        
        formulation = self._create_formulation_structure(selected_compounds, strategy, 'practical')
        formulation['reasoning'] = "Optimizing for practical implementation while maintaining good performance."
        return formulation

    def _high_performance_agent(self, balanced_compounds, specialist_compounds, strategy):
        """High-performance agent maximizing key metrics"""
        # Prefer specialist compounds for peak performance
        compounds_to_use = specialist_compounds if specialist_compounds else balanced_compounds
        if not compounds_to_use:
            return None
            
        num_components = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
        selected_compounds = random.sample(compounds_to_use, min(num_components, len(compounds_to_use)))
        
        formulation = self._create_formulation_structure(selected_compounds, strategy, 'high_performance')
        formulation['reasoning'] = "Maximizing key performance metrics through optimized component selection."
        return formulation

    def _balanced_agent(self, balanced_compounds, specialist_compounds, strategy):
        """Balanced agent seeking optimal trade-offs"""
        all_compounds = balanced_compounds + specialist_compounds
        if not all_compounds:
            return None
            
        num_components = random.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0]
        selected_compounds = random.sample(all_compounds, min(num_components, len(all_compounds)))
        
        formulation = self._create_formulation_structure(selected_compounds, strategy, 'balanced')
        formulation['reasoning'] = "Seeking optimal balance across all performance, safety, and practical criteria."
        return formulation

    def _create_formulation_structure(self, compounds: List[Dict], strategy: Dict, agent_type: str) -> Dict:
        """Create a structured formulation from selected compounds"""
        # Generate mass percentages that sum to 100
        mass_percentages = self._generate_mass_percentages(len(compounds))
        
        composition = []
        for i, compound in enumerate(compounds):
            comp_data = {
                'cid': compound.get('cid'),
                'name': compound.get('name', 'Unknown'),
                'mass_percentage': mass_percentages[i],
                'molecular_weight': compound.get('molecular_weight'),
                'smiles': compound.get('smiles'),
                'category': compound.get('category')
            }
            composition.append(comp_data)
        
        # Calculate mole ratios and percentages
        composition = self._calculate_mole_ratios(composition)
        
        return {
            'composition': composition,
            'agent_type': agent_type,
            'strategy_alignment': random.uniform(0.6, 0.95),
            'confidence': random.uniform(0.7, 0.95),
            'timestamp': np.datetime64('now')
        }

    def _generate_mass_percentages(self, num_components: int) -> List[float]:
        """Generate random mass percentages that sum to 100"""
        if num_components == 1:
            return [100.0]
        
        percentages = [random.uniform(5, 80) for _ in range(num_components)]
        total = sum(percentages)
        return [p/total * 100 for p in percentages]

    def _calculate_mole_ratios(self, composition: List[Dict]) -> List[Dict]:
        """Calculate mole ratios and percentages from mass percentages"""
        total_moles = 0
        mole_data = []
        
        for comp in composition:
            mass_percent = comp['mass_percentage']
            mw = comp.get('molecular_weight')
            
            if mw and mw > 0:
                moles = mass_percent / mw
            else:
                moles = mass_percent / 100  # Fallback
                
            mole_data.append(moles)
            total_moles += moles
        
        if total_moles > 0:
            for i, comp in enumerate(composition):
                comp['mole_percentage'] = (mole_data[i] / total_moles) * 100
                comp['mole_ratio'] = mole_data[i] / min(mole_data) if min(mole_data) > 0 else 1
            else:
            for comp in composition:
                comp['mole_percentage'] = 100.0 / len(composition)
                comp['mole_ratio'] = 1.0
        
        return composition

    def _remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations based on composition fingerprint"""
        seen = set()
        unique = []
        
        for formulation in formulations:
            # Create fingerprint based on sorted CIDs and percentages
            comps = formulation['composition']
            fingerprint = tuple(sorted(
                (comp['cid'], round(comp['mass_percentage'], 1))
                for comp in comps
            ))
            
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(formulation)
        
        return unique

    def adaptive_evaluation(self, formulations: List[Dict], strategy: Dict, 
                          negotiation_round: int, min_confidence: float) -> List[Dict]:
        """Evaluate formulations with adaptive confidence thresholds"""
        approved_formulations = []
        
        # Adaptive confidence threshold
        adaptive_confidence = min_confidence * (1.0 - (negotiation_round * 0.15))
        adaptive_confidence = max(adaptive_confidence, min_confidence * 0.5)  # Don't go below 50% of original
        
        for formulation in formulations:
            # Calculate property score
            property_score = self.calculate_property_score(formulation, strategy)
            
            # Calculate strategy alignment score
            alignment_score = formulation.get('strategy_alignment', 0.5)
            
            # Combine scores
            approval_score = (property_score * 0.6 + alignment_score * 0.4) 
            
            # Adjust for compatibility risk
            risk_score = formulation.get('compatibility_risk', 0)
            approval_score *= (1.0 - risk_score * 0.3)  # Penalize high risk
            
            formulation['approval_score'] = approval_score
            formulation['property_score'] = property_score
            
            # Enhanced AI reasoning
            formulation['reasoning'] = self._generate_evaluation_reasoning(
                formulation, strategy, property_score, risk_score
            )
            
            # Strengths and limitations
            formulation['strengths'] = self._identify_strengths(formulation, strategy)
            formulation['limitations'] = self._identify_limitations(formulation, strategy)
            
            if approval_score >= adaptive_confidence:
                approved_formulations.append(formulation)
        
        # Sort by approval score
        approved_formulations.sort(key=lambda x: x['approval_score'], reverse=True)
        return approved_formulations

    def calculate_property_score(self, formulation: Dict, strategy: Dict) -> float:
        """Calculate how well formulation matches target properties"""
        total_score = 0
        total_weight = 0
        
        target_props = strategy.get('target_properties', {})
        predicted_props = formulation.get('predicted_properties', {})
        
        for prop_name, criteria in target_props.items():
            if prop_name not in predicted_props:
                continue
                
            importance = criteria.get('importance', 'medium')
            weight = {'high': 3, 'medium': 2, 'low': 1}.get(importance, 1)
            
            predicted_value = predicted_props[prop_name]
            target_value = criteria.get('target')
            min_value = criteria.get('min')
            max_value = criteria.get('max')
            
            # Calculate individual property score
            prop_score = self._calculate_individual_property_score(
                predicted_value, target_value, min_value, max_value
            )
            
            total_score += prop_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0

    def _calculate_individual_property_score(self, predicted: float, target: float, 
                                           min_val: float, max_val: float) -> float:
        """Calculate score for individual property"""
        if min_val is not None and predicted < min_val:
            return 0.0
        if max_val is not None and predicted > max_val:
            return 0.0
            
        if target is not None:
            # Score based on proximity to target
            if target > 0:
                ratio = predicted / target
                return max(0, 1 - abs(ratio - 1) * 0.5)  # 100% score for exact match
            else:
                return 1.0 if predicted == target else 0.0
        else:
            # No specific target, just within bounds
            return 1.0

    def _generate_evaluation_reasoning(self, formulation: Dict, strategy: Dict, 
                                    property_score: float, risk_score: float) -> str:
        """Generate AI reasoning for formulation evaluation"""
        comp_names = [comp['name'] for comp in formulation['composition']]
        components_str = ", ".join(comp_names)
        
        reasoning = f"This formulation comprising {components_str} "
        
        if property_score > 0.8:
            reasoning += "demonstrates excellent alignment with target properties. "
        elif property_score > 0.6:
            reasoning += "shows good performance characteristics. "
        else:
            reasoning += "meets basic requirements but has room for improvement. "
        
        if risk_score < 0.3:
            reasoning += "The combination appears chemically stable with low compatibility risk."
        elif risk_score < 0.7:
            reasoning += "Some compatibility considerations should be reviewed during experimental validation."
        else:
            reasoning += "Significant compatibility concerns require careful handling and testing."
            
        return reasoning

    def _identify_strengths(self, formulation: Dict, strategy: Dict) -> List[str]:
        """Identify formulation strengths"""
        strengths = []
        props = formulation.get('predicted_properties', {})
        
        # Check key target properties
        target_props = strategy.get('target_properties', {})
        for prop, criteria in target_props.items():
            if prop in props:
                importance = criteria.get('importance', 'medium')
                if importance == 'high' and props[prop] >= criteria.get('min', 0):
                    strengths.append(f"Meets {prop} requirements")
        
        # Composition-based strengths
        if len(formulation['composition']) == 1:
            strengths.append("Simple single-component system")
        elif len(formulation['composition']) <= 3:
            strengths.append("Reasonable complexity for implementation")
        
        if formulation.get('compatibility_risk', 1) < 0.3:
            strengths.append("Good chemical compatibility")
            
        return strengths[:3]  # Return top 3 strengths

    def _identify_limitations(self, formulation: Dict, strategy: Dict) -> List[str]:
        """Identify formulation limitations"""
        limitations = []
        props = formulation.get('predicted_properties', {})
        
        # Check property limitations
        target_props = strategy.get('target_properties', {})
        for prop, criteria in target_props.items():
            if prop in props:
                min_val = criteria.get('min')
                max_val = criteria.get('max')
                predicted = props[prop]
                
                if min_val and predicted < min_val * 1.1:  # Close to minimum
                    limitations.append(f"Marginal {prop} performance")
                if max_val and predicted > max_val * 0.9:  # Close to maximum
                    limitations.append(f"{prop} near upper limit")
        
        # Composition-based limitations
        if len(formulation['composition']) > 4:
            limitations.append("Complex multi-component system")
        
        if formulation.get('compatibility_risk', 0) > 0.7:
            limitations.append("Potential compatibility concerns")
            
        return limitations[:3]  # Return top 3 limitations
