# ai_engine.py - ADVANCED MULTI-AGENT SYSTEM
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
            return self.get_default_strategy(material_type, negotiation_round)
        
        # Adjust prompt based on negotiation round
        if negotiation_round == 0:
            prompt_modifier = "Be very strict with requirements."
        elif negotiation_round == 1:
            prompt_modifier = "Slightly relax requirements if needed to find viable candidates."
        else:
            prompt_modifier = "Focus on finding practical solutions even if they don't meet all ideal requirements."
        
        clean_text = self.clean_challenge_text(challenge_text)
        
        prompt = f"""
        You are an expert materials scientist. Analyze this challenge and extract realistic requirements.

        CHALLENGE: {clean_text}
        MATERIAL TYPE: {material_type}
        STRATEGY: {prompt_modifier}

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
            st.warning(f"⚠️ AI interpretation failed, using default strategy")
            return self.get_default_strategy(material_type, negotiation_round)

    def get_default_strategy(self, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        # Progressive relaxation of requirements
        relaxation_factor = 1.0 - (negotiation_round * 0.15)  # 15% relaxation per round
        
        base_strategy = {
            "Coolant/Lubricant": {
                "material_class": "coolant",
                "target_properties": {
                    "thermal_conductivity": {"min": 0.1 * relaxation_factor, "target": 0.15, "unit": "W/m·K", "weight": 0.25},
                    "viscosity_100c": {"max": 10 / relaxation_factor, "target": 5, "unit": "cSt", "weight": 0.2},
                    "flash_point": {"min": 150 * relaxation_factor, "target": 200, "unit": "°C", "weight": 0.15},
                    "specific_heat": {"min": 1800 * relaxation_factor, "target": 2200, "unit": "J/kg·K", "weight": 0.15}
                },
                "safety_constraints": ["non_toxic", "pfas_free", "non_flammable"],
                "search_strategy": {
                    "chemical_classes": ["silicones", "esters", "polyalphaolefins", "glycols"],
                    "search_terms": [
                        "hexamethyldisiloxane", "octamethyltrisiloxane", "polydimethylsiloxane",
                        "mineral oil", "paraffin oil", "polyalphaolefin", "propylene glycol",
                        "ethylene glycol", "dioctyl sebacate", "dibutyl sebacate"
                    ]
                }
            }
        }
        
        return base_strategy.get(material_type, base_strategy["Coolant/Lubricant"])

    def multi_agent_formulation_generation(self, compounds_data: Dict, strategy: Dict, max_rounds: int = 5) -> List[Dict]:
        """Multi-agent system that tries different formulation strategies"""
        
        all_formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        
        st.write("🧠 Starting multi-agent formulation generation...")
        
        # Agent 1: Conservative Agent (strict requirements)
        conservative_formulations = self._conservative_agent(balanced, specialists, strategy)
        all_formulations.extend(conservative_formulations)
        st.write(f"🛡️ Conservative Agent: {len(conservative_formulations)} formulations")
        
        # Agent 2: Innovative Agent (creative combinations)
        innovative_formulations = self._innovative_agent(balanced, specialists, strategy)
        all_formulations.extend(innovative_formulations)
        st.write(f"💡 Innovative Agent: {len(innovative_formulations)} formulations")
        
        # Agent 3: Practical Agent (cost/availability focus)
        practical_formulations = self._practical_agent(balanced, specialists, strategy)
        all_formulations.extend(practical_formulations)
        st.write(f"⚙️ Practical Agent: {len(practical_formulations)} formulations")
        
        # Agent 4: Specialist Agent (property optimization)
        specialist_formulations = self._specialist_agent(balanced, specialists, strategy)
        all_formulations.extend(specialist_formulations)
        st.write(f"🎯 Specialist Agent: {len(specialist_formulations)} formulations")
        
        # Remove duplicates
        unique_formulations = self._remove_duplicate_formulations(all_formulations)
        
        st.success(f"✅ Multi-agent system generated {len(unique_formulations)} unique formulations")
        return unique_formulations

    def _conservative_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that focuses on safety and reliability"""
        formulations = []
        
        # Single compound formulations from top balanced candidates
        for i, (compound, score) in enumerate(balanced[:3]):
            formulations.append({
                'compounds': [compound],
                'ratios': [1.0],
                'strategy': 'conservative_single',
                'agent': 'conservative',
                'base_score': score,
                'risk_level': 'low'
            })
        
        return formulations

    def _innovative_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that explores creative combinations"""
        formulations = []
        
        if len(balanced) >= 2:
            # Try ternary mixtures
            for i in range(min(2, len(balanced))):
                for j in range(i+1, min(3, len(balanced))):
                    for k in range(j+1, min(4, len(balanced))):
                        comp1, score1 = balanced[i]
                        comp2, score2 = balanced[j]
                        comp3, score3 = balanced[k]
                        
                        formulations.append({
                            'compounds': [comp1, comp2, comp3],
                            'ratios': [0.5, 0.3, 0.2],
                            'strategy': 'innovative_ternary',
                            'agent': 'innovative',
                            'base_score': (score1 * 0.5 + score2 * 0.3 + score3 * 0.2),
                            'risk_level': 'medium'
                        })
        
        # Specialist combinations
        if specialists:
            top_specialists = []
            for prop_name, spec_list in list(specialists.items())[:3]:
                if spec_list:
                    top_specialists.append(spec_list[0])
            
            if len(top_specialists) >= 2:
                formulations.append({
                    'compounds': top_specialists[:2],
                    'ratios': [0.6, 0.4],
                    'strategy': 'specialist_synergy',
                    'agent': 'innovative', 
                    'base_score': 0.7,
                    'risk_level': 'medium'
                })
        
        return formulations

    def _practical_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that focuses on cost and implementation"""
        formulations = []
        
        # Simple binary mixtures
        if len(balanced) >= 2:
            for i in range(min(3, len(balanced))):
                for j in range(i+1, min(4, len(balanced))):
                    comp1, score1 = balanced[i]
                    comp2, score2 = balanced[j]
                    
                    # Try different ratios
                    for ratio in [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]:
                        formulations.append({
                            'compounds': [comp1, comp2],
                            'ratios': list(ratio),
                            'strategy': f'practical_binary_{ratio[0]}_{ratio[1]}',
                            'agent': 'practical',
                            'base_score': (score1 * ratio[0] + score2 * ratio[1]),
                            'risk_level': 'low'
                        })
        
        return formulations

    def _specialist_agent(self, balanced: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """Agent that optimizes for specific properties"""
        formulations = []
        
        if balanced and specialists:
            base_compound, base_score = balanced[0]
            
            # Enhance base compound with specialists
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
        
        return formulations

    def _remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations based on compound combinations"""
        seen = set()
        unique = []
        
        for formulation in formulations:
            # Create a unique key based on compound CIDs and ratios
            if 'compounds' in formulation:
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
                'top_formulations': [],
                'search_metrics': {'compounds_evaluated': 0, 'formulations_generated': 0, 'formulations_approved': 0},
                'strategy': strategy,
                'negotiation_round': negotiation_round
            }
        
        # Adjust confidence threshold based on negotiation round
        adaptive_confidence = min_confidence * (1.0 - (negotiation_round * 0.1))
        adaptive_confidence = max(0.3, adaptive_confidence)  # Minimum 30% confidence
        
        st.write(f"🔄 Negotiation Round {negotiation_round + 1}: Confidence threshold = {adaptive_confidence:.1%}")
        
        all_ranked = []
        approved_formulations = []
        rejected_formulations = []
        
        for formulation in formulations:
            property_score = self.calculate_property_score(formulation, strategy)
            feasibility = formulation.get('feasibility', {})
            compatibility_score = self.calculate_compatibility_score(feasibility)
            
            # Adjust scoring based on negotiation round
            negotiation_bonus = negotiation_round * 0.05
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
        
        # Return both approved and top rejected formulations
        top_rejected = rejected_formulations[:5]  # Show top 5 rejected
        
        return {
            'approved_formulations': approved_formulations[:10],
            'rejected_formulations': top_rejected,
            'all_formulations': all_ranked[:15],
            'search_metrics': {
                'compounds_evaluated': len(formulations) * 2,
                'formulations_generated': len(formulations),
                'formulations_approved': len(approved_formulations),
                'negotiation_round': negotiation_round + 1
            },
            'strategy': strategy,
            'confidence_threshold': adaptive_confidence
        }

    def calculate_property_score(self, formulation: Dict, strategy: Dict) -> float:
        total_score = 0
        total_weight = 0
        
        predicted_props = formulation.get('predicted_properties', {})
        target_props = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_props.items():
            weight = criteria.get('weight', 0.1)
            target_value = criteria.get('target')
            min_value = criteria.get('min')
            max_value = criteria.get('max')
            
            pred_value_data = predicted_props.get(prop_name, {})
            pred_value = pred_value_data.get('value', 0) if isinstance(pred_value_data, dict) else pred_value_data
            
            if target_value is not None and pred_value:
                if target_value > 0:
                    normalized_diff = abs(pred_value - target_value) / target_value
                    prop_score = max(0, 1 - normalized_diff)
                else:
                    prop_score = 0.5
            elif min_value is not None and pred_value:
                prop_score = 1.0 if pred_value >= min_value else pred_value / min_value
            elif max_value is not None and pred_value:
                prop_score = 1.0 if pred_value <= max_value else max_value / pred_value
            else:
                prop_score = 0.5
            
            total_score += prop_score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.1)

    def calculate_compatibility_score(self, feasibility: Dict) -> float:
        if not feasibility:
            return 0.7
        
        risk_score = feasibility.get('risk_score', 0.5)
        issues = feasibility.get('compatibility_issues', [])
        
        base_score = 1.0 - risk_score
        issue_penalty = len(issues) * 0.1
        
        return max(0, base_score - issue_penalty)

    def generate_ai_decision(self, formulation: Dict, overall_score: float, 
                           min_confidence: float, negotiation_round: int) -> Dict:
        approved = overall_score >= min_confidence
        
        reasons = []
        if overall_score >= 0.8:
            reasons.append("Excellent property match")
        elif overall_score >= 0.6:
            reasons.append("Good overall performance")
        elif overall_score >= 0.4:
            reasons.append("Acceptable performance with some compromises")
        else:
            reasons.append("Marginal performance - significant compromises")
        
        # Add agent-specific reasoning
        agent = formulation.get('agent', 'unknown')
        if agent == 'conservative':
            reasons.append("Conservative approach - low risk")
        elif agent == 'innovative':
            reasons.append("Innovative combination - potential high reward")
        elif agent == 'practical':
            reasons.append("Practical formulation - easy implementation")
        elif agent == 'specialist':
            reasons.append("Specialized optimization - targeted performance")
        
        compounds = formulation.get('compounds', [])
        if len(compounds) == 1:
            reasons.append("Single compound formulation")
        else:
            reasons.append(f"Multi-compound formulation ({len(compounds)} components)")
        
        if negotiation_round > 0:
            reasons.append(f"Round {negotiation_round + 1} negotiation - relaxed criteria")
        
        return {
            'approved': approved,
            'confidence': overall_score,
            'reasons': reasons,
            'recommendations': ["Recommended for experimental validation" if approved else "Consider with caution"],
            'risk_factors': formulation.get('feasibility', {}).get('compatibility_issues', [])
        }

    def clean_challenge_text(self, text: str) -> str:
        clean_text = text[:2000]
        clean_text = ' '.join(clean_text.split())
        return clean_text

    def extract_json_from_response(self, response_text: str) -> str:
        json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group()
        else:
            raise ValueError("No JSON found in AI response")

    def validate_strategy(self, strategy: Dict, material_type: str, negotiation_round: int) -> Dict:
        if 'material_class' not in strategy:
            strategy['material_class'] = material_type.lower()
        
        if 'target_properties' not in strategy:
            strategy['target_properties'] = self.get_default_strategy(material_type, negotiation_round)['target_properties']
        
        if 'search_strategy' not in strategy:
            strategy['search_strategy'] = self.get_default_strategy(material_type, negotiation_round)['search_strategy']
        
        if 'safety_constraints' not in strategy:
            strategy['safety_constraints'] = ['non_toxic', 'pfas_free']
        
        return strategy
