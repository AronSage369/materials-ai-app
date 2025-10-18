# ai_engine.py - ENHANCED WITH MOLE CALCULATIONS
import google.generativeai as genai
import json
import re
from typing import Dict, List, Any
import streamlit as st
import time
import random

class MaterialsAIEngine:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.negotiation_rounds = 0
        self.max_negotiation_rounds = 5
        
    def set_api_key(self, api_key: str):
        if not api_key:
            st.error("❌ No API key provided")
            return
            
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # FIX: Using stable model names to prevent 404 errors. 'gemini-pro' is a reliable choice.
        model_names = ['gemini-2.5-pro', 'gemini-2.5-flash']
        
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                st.sidebar.success(f"✅ Using model: {model_name}")
                return
            except Exception as e:
                continue
        
        st.error("❌ Could not initialize any Gemini model")
        self.model = None

    def interpret_challenge(self, challenge_text: str, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        if not self.model:
            st.warning("⚠️ No Gemini model initialized. Using default strategy.")
            return self.get_default_strategy(material_type, negotiation_round)
        
        # Adjust prompt based on negotiation round
        relaxation_levels = [
            "Be very strict with requirements - only accept excellent matches.",
            "Slightly relax requirements to find good candidates.",
            "Focus on finding practical solutions with minor compromises.",
            "Prioritize finding any viable solution that meets core safety requirements.",
            "Be very flexible - focus on innovative approaches over strict property matching."
        ]
        
        prompt_modifier = relaxation_levels[min(negotiation_round, len(relaxation_levels)-1)]
        
        clean_text = self.clean_challenge_text(challenge_text)
        
        prompt = f"""
        You are an expert materials scientist. Analyze this challenge and extract realistic requirements.

        CHALLENGE: {clean_text}
        MATERIAL TYPE: {material_type}
        NEGOTIATION STRATEGY: {prompt_modifier}

        Extract practical requirements as JSON with:
        - material_class
        - target_properties (with min/max/target/weight)
        - safety_constraints (must be strict)
        - search_strategy with chemical_classes and search_terms

        Focus on properties that can realistically be found in chemical databases.
        Return ONLY valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_str = self.extract_json_from_response(response.text)
            strategy = json.loads(json_str)
            strategy = self.validate_strategy(strategy, material_type, negotiation_round)
            return strategy
            
        except Exception as e:
            st.warning(f"⚠️ AI interpretation failed, using default strategy. Error: {e}")
            return self.get_default_strategy(material_type, negotiation_round)

    def get_default_strategy(self, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        # Progressive relaxation of requirements
        relaxation_factor = max(0.5, 1.0 - (negotiation_round * 0.2))  # Maximum 50% relaxation
        
        # Define base strategies for different material types
        base_strategies = {
            "Coolant/Lubricant": {
                "material_class": "coolant",
                "target_properties": {
                    "thermal_conductivity": {"min": 0.1 * relaxation_factor, "target": 0.15, "unit": "W/m·K", "weight": 0.25},
                    "viscosity_100c": {"max": 15 / relaxation_factor, "target": 8, "unit": "cSt", "weight": 0.2},
                    "flash_point": {"min": 180 * relaxation_factor, "target": 220, "unit": "°C", "weight": 0.15},
                    "specific_heat": {"min": 1800 * relaxation_factor, "target": 2200, "unit": "J/kg·K", "weight": 0.15},
                    "dielectric_strength": {"min": 30, "target": 40, "unit": "kV/2.5mm", "weight": 0.1},
                    "cost": {"max": 80 / relaxation_factor, "target": 40, "unit": "$/kg", "weight": 0.15}
                },
                "safety_constraints": ["non_toxic", "pfas_free", "non_flammable"],
            },
            "Adsorbent": {
                "material_class": "adsorbent",
                "target_properties": {
                    "surface_area": {"min": 1000 * relaxation_factor, "target": 1500, "unit": "m²/g", "weight": 0.4},
                    "pore_volume": {"min": 0.6 * relaxation_factor, "target": 0.8, "unit": "cm³/g", "weight": 0.3},
                    "co2_capacity": {"min": 2.5 * relaxation_factor, "target": 3.5, "unit": "mmol/g", "weight": 0.2},
                    "stability": {"min": 0.8, "target": 0.95, "unit": "score", "weight": 0.1}
                },
                "safety_constraints": ["non_toxic", "chemically_stable"],
            }
            # Add more material types here in the future
        }
        
        # Return the selected strategy or default to Coolant/Lubricant
        return base_strategies.get(material_type, base_strategies["Coolant/Lubricant"])

    def multi_agent_formulation_generation(self, compounds_data: Dict, strategy: Dict) -> List[Dict]:
        """Multi-agent system that tries different formulation strategies"""
        
        all_formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        
        if not balanced and not specialists:
            st.warning("⚠️ No compounds available for formulation generation.")
            return []

        st.write("🧠 Starting multi-agent formulation generation...")
        
        # Agent 1: Conservative Agent (strict requirements, simple formulations)
        conservative_formulations = self._conservative_agent(balanced, specialists, strategy)
        all_formulations.extend(conservative_formulations)
        st.write(f"🛡️ Conservative Agent: {len(conservative_formulations)} formulations")
        
        # Agent 2: Innovative Agent (complex combinations, 3-5 compounds)
        innovative_formulations = self._innovative_agent(balanced, specialists, strategy)
        all_formulations.extend(innovative_formulations)
        st.write(f"💡 Innovative Agent: {len(innovative_formulations)} formulations")
        
        # Agent 3: Practical Agent (cost/availability focus, balanced mixtures)
        practical_formulations = self._practical_agent(balanced, specialists, strategy)
        all_formulations.extend(practical_formulations)
        st.write(f"⚙️ Practical Agent: {len(practical_formulations)} formulations")
        
        # Agent 4: Specialist Agent (property optimization, enhanced mixtures)
        specialist_formulations = self._specialist_agent(balanced, specialists, strategy)
        all_formulations.extend(specialist_formulations)
        st.write(f"🎯 Specialist Agent: {len(specialist_formulations)} formulations")
        
        # Agent 5: Cocktail Agent (complex multi-compound mixtures)
        cocktail_formulations = self._cocktail_agent(balanced, specialists, strategy)
        all_formulations.extend(cocktail_formulations)
        st.write(f"🍹 Cocktail Agent: {len(cocktail_formulations)} formulations")
        
        # Remove duplicates
        unique_formulations = self._remove_duplicate_formulations(all_formulations)
        
        st.success(f"✅ Multi-agent system generated {len(unique_formulations)} unique formulations")
        return unique_formulations

    def _conservative_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that focuses on safety and reliability - simple formulations"""
        formulations = []
        
        # Single compound formulations from top balanced candidates
        for i, (compound, score) in enumerate(balanced[:5]):
            formulations.append({
                'compounds': [compound],
                'ratios': [1.0],
                'strategy': 'conservative_single',
                'agent': 'conservative',
                'base_score': score,
                'risk_level': 'low'
            })
        
        # Simple binary mixtures
        if len(balanced) >= 2:
            for i in range(min(3, len(balanced))):
                for j in range(i+1, min(4, len(balanced))):
                    comp1, score1 = balanced[i]
                    comp2, score2 = balanced[j]
                    
                    formulations.append({
                        'compounds': [comp1, comp2],
                        'ratios': [0.7, 0.3],
                        'strategy': 'conservative_binary',
                        'agent': 'conservative',
                        'base_score': (score1 * 0.7 + score2 * 0.3),
                        'risk_level': 'low'
                    })
        
        return formulations

    def _innovative_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that explores complex combinations (3-5 compounds)"""
        formulations = []
        
        # Complex ternary mixtures (3 compounds)
        if len(balanced) >= 3:
            for i in range(min(3, len(balanced))):
                for j in range(i+1, min(4, len(balanced))):
                    for k in range(j+1, min(5, len(balanced))):
                        comp1, score1 = balanced[i]
                        comp2, score2 = balanced[j]
                        comp3, score3 = balanced[k]
                        
                        # Try different ratio patterns
                        ratio_patterns = [
                            [0.5, 0.3, 0.2],  # Dominant base
                            [0.4, 0.35, 0.25],  # Balanced
                            [0.6, 0.25, 0.15],  # Strong base
                        ]
                        
                        for ratios in ratio_patterns:
                            formulations.append({
                                'compounds': [comp1, comp2, comp3],
                                'ratios': ratios,
                                'strategy': 'innovative_ternary',
                                'agent': 'innovative',
                                'base_score': (score1 * ratios[0] + score2 * ratios[1] + score3 * ratios[2]),
                                'risk_level': 'medium'
                            })
        
        # Quaternary mixtures (4 compounds)
        if len(balanced) >= 4:
            for i in range(min(2, len(balanced))):
                for j in range(i+1, min(3, len(balanced))):
                    for k in range(j+1, min(4, len(balanced))):
                        for l in range(k+1, min(5, len(balanced))):
                            compounds = [balanced[i][0], balanced[j][0], balanced[k][0], balanced[l][0]]
                            scores = [balanced[i][1], balanced[j][1], balanced[k][1], balanced[l][1]]
                            
                            ratios = [0.4, 0.25, 0.2, 0.15]  # Base + modifiers
                            total_score = sum(s * r for s, r in zip(scores, ratios))
                            
                            formulations.append({
                                'compounds': compounds,
                                'ratios': ratios,
                                'strategy': 'innovative_quaternary',
                                'agent': 'innovative',
                                'base_score': total_score,
                                'risk_level': 'high'
                            })
        
        # Quinary mixtures (5 compounds) - maximum complexity
        if len(balanced) >= 5:
            top_compounds = [c[0] for c in balanced[:5]]
            top_scores = [c[1] for c in balanced[:5]]
            
            ratios = [0.3, 0.25, 0.2, 0.15, 0.1]  # Graduated composition
            total_score = sum(s * r for s, r in zip(top_scores, ratios))
            
            formulations.append({
                'compounds': top_compounds,
                'ratios': ratios,
                'strategy': 'innovative_quinary',
                'agent': 'innovative',
                'base_score': total_score,
                'risk_level': 'high'
            })
        
        return formulations

    def _practical_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that focuses on cost and implementation - balanced mixtures"""
        formulations = []
        
        # Binary mixtures with various ratios
        if len(balanced) >= 2:
            ratio_combinations = [
                [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5],
                [0.9, 0.1], [0.65, 0.35]
            ]
            
            for i in range(min(4, len(balanced))):
                for j in range(i+1, min(5, len(balanced))):
                    comp1, score1 = balanced[i]
                    comp2, score2 = balanced[j]
                    
                    for ratios in ratio_combinations:
                        formulations.append({
                            'compounds': [comp1, comp2],
                            'ratios': ratios,
                            'strategy': f'practical_binary_{ratios[0]}_{ratios[1]}',
                            'agent': 'practical',
                            'base_score': (score1 * ratios[0] + score2 * ratios[1]),
                            'risk_level': 'low'
                        })
        
        # Ternary practical mixtures
        if len(balanced) >= 3:
            for i in range(min(3, len(balanced))):
                for j in range(i+1, min(4, len(balanced))):
                    for k in range(j+1, min(5, len(balanced))):
                        comp1, score1 = balanced[i]
                        comp2, score2 = balanced[j]
                        comp3, score3 = balanced[k]
                        
                        ratios = [0.6, 0.25, 0.15]  # Practical ternary
                        total_score = (score1 * ratios[0] + score2 * ratios[1] + score3 * ratios[2])
                        
                        formulations.append({
                            'compounds': [comp1, comp2, comp3],
                            'ratios': ratios,
                            'strategy': 'practical_ternary',
                            'agent': 'practical',
                            'base_score': total_score,
                            'risk_level': 'medium'
                        })
        
        return formulations

    def _specialist_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that optimizes for specific properties using specialists"""
        formulations = []
        
        if balanced and specialists:
            base_compound, base_score = balanced[0]
            
            # Base + 1 specialist
            for prop_name, spec_list in specialists.items():
                if spec_list:
                    specialist = spec_list[0]
                    
                    formulations.append({
                        'compounds': [base_compound, specialist],
                        'ratios': [0.7, 0.3],
                        'strategy': f'enhanced_{prop_name}',
                        'agent': 'specialist',
                        'base_score': base_score * 0.7 + 0.3,
                        'risk_level': 'medium'
                    })
            
            # Base + 2 specialists (ternary)
            if len(specialists) >= 2:
                spec_props = list(specialists.keys())[:2]
                specialists_compounds = [specialists[prop][0] for prop in spec_props if specialists.get(prop)]
                
                if len(specialists_compounds) == 2:
                    formulations.append({
                        'compounds': [base_compound, specialists_compounds[0], specialists_compounds[1]],
                        'ratios': [0.6, 0.25, 0.15],
                        'strategy': f'enhanced_{spec_props[0]}_{spec_props[1]}',
                        'agent': 'specialist',
                        'base_score': base_score * 0.6 + 0.4,
                        'risk_level': 'medium'
                    })
            
            # Base + 3 specialists (quaternary)
            if len(specialists) >= 3:
                spec_props = list(specialists.keys())[:3]
                specialists_compounds = [specialists[prop][0] for prop in spec_props if specialists.get(prop)]
                
                if len(specialists_compounds) == 3:
                    formulations.append({
                        'compounds': [base_compound, *specialists_compounds],
                        'ratios': [0.5, 0.2, 0.2, 0.1],
                        'strategy': f'enhanced_triple_specialist',
                        'agent': 'specialist',
                        'base_score': base_score * 0.5 + 0.5,
                        'risk_level': 'high'
                    })
        
        return formulations

    def _cocktail_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that creates complex cocktail mixtures"""
        formulations = []
        
        if len(balanced) >= 5:
            # Create complex cocktails with 5 compounds
            top_compounds = [c[0] for c in balanced[:5]]
            top_scores = [c[1] for c in balanced[:5]]
            
            # Different cocktail recipes
            cocktail_recipes = [
                [0.35, 0.25, 0.2, 0.15, 0.05],  # Graduated
                [0.4, 0.2, 0.2, 0.1, 0.1],      # Balanced
                [0.5, 0.15, 0.15, 0.1, 0.1],    # Strong base
            ]
            
            for recipe in cocktail_recipes:
                if len(top_compounds) >= len(recipe):
                    compounds = top_compounds[:len(recipe)]
                    scores = top_scores[:len(recipe)]
                    total_score = sum(s * r for s, r in zip(scores, recipe))
                    
                    formulations.append({
                        'compounds': compounds,
                        'ratios': recipe,
                        'strategy': 'cocktail_complex',
                        'agent': 'cocktail',
                        'base_score': total_score,
                        'risk_level': 'high'
                    })
        
        return formulations

    def calculate_mole_percentages(self, formulations: List[Dict]) -> List[Dict]:
        """Calculate mole percentages for all formulations"""
        for formulation in formulations:
            compounds = formulation.get('compounds', [])
            mass_ratios = formulation.get('ratios', [])
            
            if len(compounds) > 1 and len(mass_ratios) == len(compounds):
                mole_ratios = self._calculate_mole_ratios(compounds, mass_ratios)
                formulation['mole_ratios'] = mole_ratios
        
        return formulations

    def _calculate_mole_ratios(self, compounds: List, mass_ratios: List[float]) -> List[float]:
        """Calculate mole ratios from mass ratios and molecular weights"""
        try:
            moles = []
            total_moles = 0.0
            
            for i, compound in enumerate(compounds):
                if (hasattr(compound, 'molecular_weight') and 
                    compound.molecular_weight and 
                    float(compound.molecular_weight) > 0):
                    
                    # moles = mass / molecular_weight
                    mole_amount = mass_ratios[i] / float(compound.molecular_weight)
                    moles.append(mole_amount)
                    total_moles += mole_amount
                else:
                    # If molecular weight not available, cannot calculate moles accurately
                    return [] # Return empty list to indicate failure
            
            # Normalize to get mole fractions
            if total_moles > 0:
                mole_fractions = [mole / total_moles for mole in moles]
                return mole_fractions
            else:
                return []
            
        except Exception as e:
            st.write(f"⚠️ Error calculating mole ratios: {e}")
            return []  # Return empty on error

    def _remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations based on compound combinations"""
        seen = set()
        unique = []
        
        for formulation in formulations:
            if 'compounds' in formulation:
                # Create a key from sorted CIDs and ratios
                cids = tuple(sorted([getattr(c, 'cid', 'unknown') for c in formulation['compounds']]))
                ratios_key = tuple(formulation.get('ratios', []))
                key = (cids, ratios_key)
                
                if key not in seen:
                    seen.add(key)
                    unique.append(formulation)
        
        return unique

    def adaptive_evaluation(self, formulations: List[Dict], strategy: Dict, 
                          min_confidence: float, negotiation_round: int) -> Dict[str, Any]:
        """Adaptive evaluation that becomes more lenient each round"""
        
        if not formulations:
            return {
                'approved_formulations': [], 'rejected_formulations': [],
                'search_metrics': {}, 'strategy': strategy,
                'negotiation_round': negotiation_round
            }
        
        # Adjust confidence threshold based on negotiation round
        adaptive_confidence = min_confidence * (1.0 - (negotiation_round * 0.15))
        adaptive_confidence = max(0.3, adaptive_confidence)  # Minimum 30% confidence
        
        st.write(f"🔄 Negotiation Round {negotiation_round + 1}: Confidence threshold = {adaptive_confidence:.1%}")
        
        all_ranked = []
        approved_formulations = []
        rejected_formulations = []
        
        for formulation in formulations:
            property_score = self.calculate_property_score(formulation, strategy)
            feasibility = formulation.get('feasibility', {})
            compatibility_score = self.calculate_compatibility_score(feasibility)
            
            negotiation_bonus = negotiation_round * 0.08
            overall_score = min(1.0, (property_score * 0.7 + compatibility_score * 0.3) + negotiation_bonus)
            
            ai_decision = self.generate_ai_decision(formulation, overall_score, adaptive_confidence, negotiation_round)
            
            formulation['score'] = overall_score
            formulation['ai_decision'] = ai_decision
            formulation['property_score'] = property_score
            formulation['compatibility_score'] = compatibility_score
            formulation['negotiation_round'] = negotiation_round
            
            all_ranked.append(formulation)
            
            if ai_decision['approved']:
                approved_formulations.append(formulation)
            else:
                rejected_formulations.append(formulation)
        
        # Sort all formulations by score
        all_ranked.sort(key=lambda x: x['score'], reverse=True)
        approved_formulations.sort(key=lambda x: x['score'], reverse=True)
        rejected_formulations.sort(key=lambda x: x['score'], reverse=True)
        
        # FIX: Return ALL approved/rejected formulations. Slicing will be handled in app.py
        return {
            'approved_formulations': approved_formulations,
            'rejected_formulations': rejected_formulations,
            'search_metrics': {
                'compounds_evaluated': len(formulations) * 2,
                'formulations_generated': len(formulations),
                'formulations_approved': len(approved_formulations),
            },
            'strategy': strategy,
            'confidence_threshold': adaptive_confidence
        }

    def calculate_property_score(self, formulation: Dict, strategy: Dict) -> float:
        total_score = 0
        total_weight = 0
        
        predicted_props = formulation.get('predicted_properties', {})
        target_props = strategy.get('target_properties', {})
        
        if not isinstance(target_props, dict): return 0.5

        for prop_name, criteria in target_props.items():
            weight = criteria.get('weight', 0.1)
            target_value = criteria.get('target')
            min_value = criteria.get('min')
            max_value = criteria.get('max')
            
            pred_value_data = predicted_props.get(prop_name, {})
            pred_value = pred_value_data.get('value', 0) if isinstance(pred_value_data, dict) else pred_value_data
            
            if not isinstance(pred_value, (int, float)): continue

            prop_score = 0.5
            if target_value is not None and target_value > 0:
                normalized_diff = abs(pred_value - target_value) / target_value
                prop_score = max(0, 1 - normalized_diff)
            elif min_value is not None and pred_value >= min_value:
                prop_score = 1.0
            elif min_value is not None and pred_value < min_value:
                prop_score = max(0, pred_value / min_value)
            elif max_value is not None and pred_value <= max_value:
                prop_score = 1.0
            elif max_value is not None and pred_value > max_value:
                prop_score = max(0, max_value / pred_value)
            
            total_score += prop_score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.1)

    def calculate_compatibility_score(self, feasibility: Dict) -> float:
        if not feasibility: return 0.7
        risk_score = feasibility.get('risk_score', 0.5)
        issues = feasibility.get('compatibility_issues', [])
        base_score = 1.0 - risk_score
        issue_penalty = len(issues) * 0.1
        return max(0, base_score - issue_penalty)

    def generate_ai_decision(self, formulation: Dict, overall_score: float, 
                           min_confidence: float, negotiation_round: int) -> Dict:
        approved = overall_score >= min_confidence
        reasons = []
        
        if overall_score >= 0.8: reasons.append("Excellent property match with requirements")
        elif overall_score >= 0.6: reasons.append("Good performance with minor compromises")
        else: reasons.append("Acceptable performance with some trade-offs")
        
        agent = formulation.get('agent', 'unknown')
        if agent == 'conservative': reasons.append("Conservative approach - low implementation risk")
        elif agent == 'innovative': reasons.append("Innovative formulation - potential high reward")
        
        compound_count = len(formulation.get('compounds', []))
        if compound_count == 1: reasons.append("Single compound - simplest implementation")
        elif compound_count == 2: reasons.append("Binary mixture - good balance of simplicity and performance")
        else: reasons.append(f"Complex mixture ({compound_count} compounds) - maximum optimization potential")
        
        if negotiation_round > 0: reasons.append(f"Round {negotiation_round + 1} - relaxed evaluation criteria")
        
        return {
            'approved': approved, 'confidence': overall_score,
            'reasons': reasons,
            'recommendations': ["Strongly recommended for experimental validation" if approved else "Consider with caution"],
            'risk_factors': formulation.get('feasibility', {}).get('compatibility_issues', [])
        }

    def clean_challenge_text(self, text: str) -> str:
        clean_text = text[:2000]
        clean_text = ' '.join(clean_text.split())
        return clean_text

    def extract_json_from_response(self, response_text: str) -> str:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return response_text[start_index:end_index+1]
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match: return json_match.group()
        else: raise ValueError("No JSON found in AI response")

    def validate_strategy(self, strategy: Dict, material_type: str, negotiation_round: int) -> Dict:
        if 'material_class' not in strategy:
            strategy['material_class'] = material_type.lower()
        if 'target_properties' not in strategy:
            strategy['target_properties'] = self.get_default_strategy(material_type, negotiation_round)['target_properties']
        if 'search_strategy' not in strategy:
            default_strategy = self.get_default_strategy(material_type, negotiation_round)
            strategy['search_strategy'] = default_strategy.get('search_strategy', {})
        if 'safety_constraints' not in strategy:
            strategy['safety_constraints'] = ['non_toxic', 'pfas_free']
        return strategy

