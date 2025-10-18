# ai_engine.py - ENHANCED WITH MOLE CALCULATIONS AND IMPROVED AGENT LOGIC
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
        
        # Using the latest powerful and reliable models
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

        Extract practical requirements as a single, valid JSON object with these exact keys:
        - "material_class": (string)
        - "target_properties": (object with properties as keys, and objects with min/max/target/weight as values)
        - "safety_constraints": (array of strings)
        - "search_strategy": (object with "chemical_classes" and "search_terms" as keys)

        Focus on properties that can realistically be found in chemical databases.
        Return ONLY the raw JSON object and nothing else.
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_str = self.extract_json_from_response(response.text)
            strategy = json.loads(json_str)
            
            # CRITICAL FIX: Ensure the loaded strategy is a dictionary before validation
            if not isinstance(strategy, dict):
                raise ValueError("AI response did not resolve to a dictionary.")
                
            strategy = self.validate_strategy(strategy, material_type, negotiation_round)
            return strategy
            
        except Exception as e:
            st.warning(f"⚠️ AI interpretation failed, using default strategy. Error: {e}")
            return self.get_default_strategy(material_type, negotiation_round)

    def get_default_strategy(self, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        relaxation_factor = max(0.5, 1.0 - (negotiation_round * 0.2))
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
        }
        return base_strategies.get(material_type, base_strategies["Coolant/Lubricant"])

    def multi_agent_formulation_generation(self, compounds_data: Dict, strategy: Dict) -> List[Dict]:
        all_formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        
        if not balanced and not any(specialists.values()):
            st.warning("⚠️ No compounds available for formulation generation.")
            return []

        # IMPROVEMENT: Create a wider pool of candidates for agents to be more creative
        candidate_pool = balanced[:25] # Use top 25 balanced compounds
        if not candidate_pool:
            st.warning("⚠️ Candidate pool is empty; agents will have limited creativity.")

        st.write("🧠 Starting multi-agent formulation generation...")
        
        agent_functions = {
            "Conservative": self._conservative_agent,
            "Innovative": self._innovative_agent,
            "Practical": self._practical_agent,
            "Specialist": self._specialist_agent,
            "Cocktail": self._cocktail_agent
        }
        
        agent_icons = {"Conservative": "🛡️", "Innovative": "💡", "Practical": "⚙️", "Specialist": "🎯", "Cocktail": "🍹"}

        for name, func in agent_functions.items():
            formulations = func(candidate_pool, specialists, strategy)
            all_formulations.extend(formulations)
            st.write(f"{agent_icons[name]} {name} Agent: {len(formulations)} formulations")
        
        unique_formulations = self._remove_duplicate_formulations(all_formulations)
        st.success(f"✅ Multi-agent system generated {len(unique_formulations)} unique formulations")
        return unique_formulations

    def _conservative_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if not pool: return formulations

        # Top 5 single compound formulations
        for compound, score in pool[:5]:
            formulations.append({'compounds': [compound], 'ratios': [1.0], 'agent': 'Conservative', 'risk_level': 'Low'})
        
        # Simple binary mixtures from top 5
        if len(pool) >= 2:
            for i in range(min(5, len(pool))):
                for j in range(i + 1, min(5, len(pool))):
                    comp1, _ = pool[i]
                    comp2, _ = pool[j]
                    formulations.append({'compounds': [comp1, comp2], 'ratios': [0.7, 0.3], 'agent': 'Conservative', 'risk_level': 'Low'})
        return formulations

    def _innovative_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """IMPROVED: Uses random sampling for more diverse, complex mixtures."""
        formulations = []
        if len(pool) < 3: return formulations
        
        # Generate 20 diverse ternary mixtures
        for _ in range(20):
            if len(pool) < 3: break
            try:
                comps = random.sample(pool, 3)
                formulations.append({'compounds': [c[0] for c in comps], 'ratios': [0.5, 0.3, 0.2], 'agent': 'Innovative', 'risk_level': 'Medium'})
            except ValueError: continue
        
        # Generate 10 diverse quaternary mixtures
        for _ in range(10):
            if len(pool) < 4: break
            try:
                comps = random.sample(pool, 4)
                formulations.append({'compounds': [c[0] for c in comps], 'ratios': [0.4, 0.3, 0.2, 0.1], 'agent': 'Innovative', 'risk_level': 'High'})
            except ValueError: continue
            
        return formulations

    def _practical_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """IMPROVED: Generates a wider range of practical binary and ternary ratios."""
        formulations = []
        if len(pool) < 2: return formulations
        
        ratio_combinations = [[0.8, 0.2], [0.6, 0.4], [0.5, 0.5], [0.9, 0.1]]
        
        # Create binary mixtures from top 7 candidates
        for i in range(min(7, len(pool))):
            for j in range(i + 1, min(7, len(pool))):
                for r in ratio_combinations:
                    formulations.append({'compounds': [pool[i][0], pool[j][0]], 'ratios': r, 'agent': 'Practical', 'risk_level': 'Low'})
        
        # Simple ternary mixtures
        if len(pool) >= 3:
            for _ in range(10):
                try:
                    comps = random.sample(pool, 3)
                    formulations.append({'compounds': [c[0] for c in comps], 'ratios': [0.6, 0.25, 0.15], 'agent': 'Practical', 'risk_level': 'Medium'})
                except ValueError: continue
                
        return formulations

    def _specialist_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """IMPROVED: Methodically combines top candidates with various specialists."""
        formulations = []
        if not pool or not specialists: return formulations

        # Combine top 3 balanced performers with top 2 specialists from each category
        for base_compound, _ in pool[:3]:
            for prop_name, spec_list in specialists.items():
                for specialist_compound in spec_list[:2]:
                    if getattr(base_compound, 'cid', 0) == getattr(specialist_compound, 'cid', 1): continue
                    formulations.append({'compounds': [base_compound, specialist_compound], 'ratios': [0.85, 0.15], 'agent': 'Specialist', 'risk_level': 'Medium'})
        return formulations

    def _cocktail_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        """IMPROVED: Creates complex cocktails blending balanced performers and specialists."""
        formulations = []
        if len(pool) < 3: return formulations
        
        all_specialists = [s for sublist in specialists.values() for s in sublist]
        if not all_specialists: return formulations
        
        # Create 15 cocktails of 2 balanced + 1 specialist
        for _ in range(15):
            try:
                base_comps = [c[0] for c in random.sample(pool, 2)]
                spec_comp = random.choice(all_specialists)
                formulations.append({'compounds': [*base_comps, spec_comp], 'ratios': [0.4, 0.4, 0.2], 'agent': 'Cocktail', 'risk_level': 'High'})
            except (ValueError, IndexError): continue
        
        return formulations
    
    # ... [Rest of the file remains largely the same] ...
    # CRITICAL FIX in validate_strategy
    def validate_strategy(self, strategy: Dict, material_type: str, negotiation_round: int) -> Dict:
        # This function now ensures that key parts of the strategy are of the correct type.
        
        if 'material_class' not in strategy or not isinstance(strategy.get('material_class'), str):
            strategy['material_class'] = material_type.lower().split('/')[0]

        if 'target_properties' not in strategy or not isinstance(strategy.get('target_properties'), dict):
            st.warning("⚠️ AI returned invalid 'target_properties'. Reverting to default.")
            strategy['target_properties'] = self.get_default_strategy(material_type, negotiation_round)['target_properties']
        
        if 'search_strategy' not in strategy or not isinstance(strategy.get('search_strategy'), dict):
            strategy['search_strategy'] = self.get_default_strategy(material_type, negotiation_round).get('search_strategy', {})
        
        if 'safety_constraints' not in strategy or not isinstance(strategy.get('safety_constraints'), list):
            strategy['safety_constraints'] = ['non_toxic', 'pfas_free']
            
        return strategy
    # ... [Other functions like calculate_mole_percentages, adaptive_evaluation, etc.] ...

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
                    
                    mole_amount = mass_ratios[i] / float(compound.molecular_weight)
                    moles.append(mole_amount)
                    total_moles += mole_amount
                else:
                    return [] 
            
            if total_moles > 0:
                mole_fractions = [mole / total_moles for mole in moles]
                return mole_fractions
            else:
                return []
            
        except Exception as e:
            st.write(f"⚠️ Error calculating mole ratios: {e}")
            return []

    def _remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations based on compound combinations"""
        seen = set()
        unique = []
        
        for formulation in formulations:
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
        if not formulations:
            return {'approved_formulations': [], 'rejected_formulations': [], 'search_metrics': {}, 'strategy': strategy, 'negotiation_round': negotiation_round}
        
        adaptive_confidence = max(0.3, min_confidence * (1.0 - (negotiation_round * 0.15)))
        st.write(f"🔄 Negotiation Round {negotiation_round + 1}: Confidence threshold = {adaptive_confidence:.1%}")
        
        approved_formulations, rejected_formulations = [], []
        
        for formulation in formulations:
            property_score = self.calculate_property_score(formulation, strategy)
            compatibility_score = self.calculate_compatibility_score(formulation.get('feasibility', {}))
            negotiation_bonus = negotiation_round * 0.08
            overall_score = min(1.0, (property_score * 0.7 + compatibility_score * 0.3) + negotiation_bonus)
            
            ai_decision = self.generate_ai_decision(formulation, overall_score, adaptive_confidence, negotiation_round)
            formulation.update({
                'score': overall_score, 'ai_decision': ai_decision,
                'property_score': property_score, 'compatibility_score': compatibility_score,
                'negotiation_round': negotiation_round
            })
            
            (approved_formulations if ai_decision['approved'] else rejected_formulations).append(formulation)

        approved_formulations.sort(key=lambda x: x['score'], reverse=True)
        rejected_formulations.sort(key=lambda x: x['score'], reverse=True)
        
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
        total_score, total_weight = 0, 0
        predicted_props = formulation.get('predicted_properties', {})
        target_props = strategy.get('target_properties', {})
        
        if not isinstance(target_props, dict): return 0.5

        for prop_name, criteria in target_props.items():
            weight = criteria.get('weight', 0.1)
            pred_value_data = predicted_props.get(prop_name, {})
            pred_value = pred_value_data.get('value', 0) if isinstance(pred_value_data, dict) else pred_value_data
            
            if not isinstance(pred_value, (int, float)): continue
            
            prop_score = self._score_property(pred_value, criteria)
            total_score += prop_score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.1)

    def _score_property(self, pred_value, criteria):
        target = criteria.get('target')
        min_val = criteria.get('min')
        max_val = criteria.get('max')
        
        if target is not None and target > 0:
            return max(0, 1 - abs(pred_value - target) / target)
        elif min_val is not None:
            return 1.0 if pred_value >= min_val else max(0, pred_value / min_val)
        elif max_val is not None:
            return 1.0 if pred_value <= max_val else max(0, max_val / pred_value)
        return 0.5

    def calculate_compatibility_score(self, feasibility: Dict) -> float:
        if not feasibility: return 0.7
        risk_score = feasibility.get('risk_score', 0.5)
        base_score = 1.0 - risk_score
        issue_penalty = len(feasibility.get('compatibility_issues', [])) * 0.1
        return max(0, base_score - issue_penalty)

    def generate_ai_decision(self, formulation: Dict, overall_score: float, min_confidence: float, negotiation_round: int) -> Dict:
        approved = overall_score >= min_confidence
        reasons = []
        
        if overall_score >= 0.8: reasons.append("Excellent property match")
        elif overall_score >= 0.6: reasons.append("Good performance with compromises")
        else: reasons.append("Acceptable performance with trade-offs")
        
        reasons.append(f"{formulation.get('agent', 'Unknown')} Agent strategy")
        
        if negotiation_round > 0: reasons.append(f"Found in relaxed Round {negotiation_round + 1}")
        
        return {'approved': approved, 'confidence': overall_score, 'reasons': reasons}

    def clean_challenge_text(self, text: str) -> str:
        return ' '.join(text[:2000].split())

    def extract_json_from_response(self, response_text: str) -> str:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match: return match.group()
        raise ValueError("No JSON found in AI response")

