# ai_engine.py - ENHANCED WITH SAFE MOLECULAR WEIGHT HANDLING
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
        
    def set_api_key(self, api_key: str):
        if not api_key: return
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        model_names = ['gemini-2.5-pro', 'gemini-2.5-flash']
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                return
            except Exception:
                continue
        st.error("❌ Could not initialize any Gemini model.")
        self.model = None

    def interpret_challenge(self, challenge_text: str, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        if not self.model: return self.get_default_strategy(material_type, negotiation_round)
        # ... [rest of implementation unchanged] ...
        relaxation_levels = [
            "strict requirements, excellent matches only.", "slightly relaxed requirements for good candidates.",
            "focus on practical solutions with minor compromises.", "prioritize any viable solution meeting core safety needs.",
            "be very flexible, focus on innovation over strict matching."
        ]
        prompt_modifier = relaxation_levels[min(negotiation_round, len(relaxation_levels)-1)]
        prompt = f"""
        Analyze the materials challenge. Return a single, valid JSON object.
        CHALLENGE: "{self.clean_challenge_text(challenge_text)}"
        MATERIAL TYPE: "{material_type}" STRATEGY: "{prompt_modifier}"
        JSON STRUCTURE: {{"material_class": str, "target_properties": {{prop: {{"min": float, "max": float, "target": float, "weight": float}}}}, "safety_constraints": [str], "search_strategy": {{"search_terms": [str]}}}}
        RESPONSE:
        """
        try:
            response = self.model.generate_content(prompt)
            strategy = json.loads(self.extract_json_from_response(response.text))
            if not isinstance(strategy, dict): raise ValueError("Response is not a dictionary.")
            return self.validate_strategy(strategy, material_type, negotiation_round)
        except Exception as e:
            st.toast(f"⚠️ AI interpretation failed. Using default. Error: {e}", icon="🤖")
            return self.get_default_strategy(material_type, negotiation_round)

    def get_default_strategy(self, material_type: str, negotiation_round: int = 0) -> Dict[str, Any]:
        # ... [Implementation remains unchanged] ...
        relaxation_factor = max(0.5, 1.0 - (negotiation_round * 0.2))
        base_strategies = {
            "Coolant/Lubricant": { "material_class": "coolant", "target_properties": { "thermal_conductivity": {"min": 0.1 * relaxation_factor, "target": 0.15, "unit": "W/m·K", "weight": 0.25}, "viscosity_100c": {"max": 15 / relaxation_factor, "target": 8, "unit": "cSt", "weight": 0.2}, "flash_point": {"min": 180 * relaxation_factor, "target": 220, "unit": "°C", "weight": 0.15}, }, "safety_constraints": ["non_toxic", "pfas_free"], },
            "Adsorbent": { "material_class": "adsorbent", "target_properties": { "surface_area": {"min": 1000 * relaxation_factor, "target": 1500, "unit": "m²/g", "weight": 0.4}, "pore_volume": {"min": 0.6 * relaxation_factor, "target": 0.8, "unit": "cm³/g", "weight": 0.3}, }, "safety_constraints": ["non_toxic", "chemically_stable"], }
        }
        return base_strategies.get(material_type, base_strategies["Coolant/Lubricant"])

    def multi_agent_formulation_generation(self, compounds_data: Dict, strategy: Dict) -> List[Dict]:
        # ... [Implementation with enhanced random sampling remains unchanged] ...
        all_formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        if not balanced and not any(specialists.values()): return []
        candidate_pool = balanced[:25]
        agent_functions = { "Conservative": self._conservative_agent, "Innovative": self._innovative_agent, "Practical": self._practical_agent, "Specialist": self._specialist_agent, "Cocktail": self._cocktail_agent }
        for name, func in agent_functions.items():
            formulations = func(candidate_pool, specialists, strategy)
            all_formulations.extend(formulations)
        return self._remove_duplicate_formulations(all_formulations)

    # All agent implementations (_conservative_agent, etc.) remain the same.
    def _conservative_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if not pool: return formulations
        for compound, score in pool[:5]:
            formulations.append({'compounds': [compound], 'ratios': [1.0], 'agent': 'Conservative', 'risk_level': 'Low'})
        if len(pool) >= 2:
            for i in range(min(5, len(pool))):
                for j in range(i + 1, min(5, len(pool))):
                    formulations.append({'compounds': [pool[i][0], pool[j][0]], 'ratios': [0.7, 0.3], 'agent': 'Conservative', 'risk_level': 'Low'})
        return formulations
    def _innovative_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if len(pool) < 3: return formulations
        for _ in range(20):
            if len(pool) < 3: break
            try:
                comps = random.sample(pool, 3)
                formulations.append({'compounds': [c[0] for c in comps], 'ratios': [0.5, 0.3, 0.2], 'agent': 'Innovative', 'risk_level': 'Medium'})
            except ValueError: continue
        return formulations
    def _practical_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if len(pool) < 2: return formulations
        for i in range(min(7, len(pool))):
            for j in range(i + 1, min(7, len(pool))):
                for r in [[0.8, 0.2], [0.5, 0.5]]:
                    formulations.append({'compounds': [pool[i][0], pool[j][0]], 'ratios': r, 'agent': 'Practical', 'risk_level': 'Low'})
        return formulations
    def _specialist_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if not pool or not specialists: return formulations
        for base_compound, _ in pool[:3]:
            for prop_name, spec_list in specialists.items():
                for specialist_compound in spec_list[:2]:
                    if getattr(base_compound, 'cid', 0) != getattr(specialist_compound, 'cid', 1):
                        formulations.append({'compounds': [base_compound, specialist_compound], 'ratios': [0.85, 0.15], 'agent': 'Specialist', 'risk_level': 'Medium'})
        return formulations
    def _cocktail_agent(self, pool: List, specialists: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        if len(pool) < 2: return formulations
        all_specialists = [s for sublist in specialists.values() for s in sublist]
        if not all_specialists: return formulations
        for _ in range(15):
            try:
                base_comp = random.choice(pool)[0]
                spec_comp = random.choice(all_specialists)
                formulations.append({'compounds': [base_comp, spec_comp], 'ratios': [0.7, 0.3], 'agent': 'Cocktail', 'risk_level': 'High'})
            except (ValueError, IndexError): continue
        return formulations

    def validate_strategy(self, strategy: Dict, material_type: str, negotiation_round: int) -> Dict:
        if 'target_properties' not in strategy or not isinstance(strategy.get('target_properties'), dict):
            strategy['target_properties'] = self.get_default_strategy(material_type, negotiation_round)['target_properties']
        return strategy

    def calculate_mole_percentages(self, formulations: List[Dict]) -> List[Dict]:
        for f in formulations:
            if len(f.get('compounds', [])) > 1:
                f['mole_ratios'] = self._calculate_mole_ratios(f['compounds'], f['ratios'])
        return formulations

    # --- FIX: ADDED SAFE MOLECULAR WEIGHT HELPER ---
    def _get_safe_mw(self, compound) -> float:
        """Safely gets molecular weight as a float, returning 0.0 on failure."""
        try:
            return float(getattr(compound, 'molecular_weight', 0.0))
        except (ValueError, TypeError):
            return 0.0

    def _calculate_mole_ratios(self, compounds: List, mass_ratios: List[float]) -> List[float]:
        """Calculates mole ratios using the safe MW helper function."""
        try:
            moles, total_moles = [], 0.0
            for i, compound in enumerate(compounds):
                mw = self._get_safe_mw(compound) # Using the safe helper here
                if mw > 0:
                    mole_amount = mass_ratios[i] / mw
                    moles.append(mole_amount)
                    total_moles += mole_amount
                else:
                    return [] # Cannot calculate if any MW is invalid
            
            return [mole / total_moles for mole in moles] if total_moles > 0 else []
        except Exception:
            return []

    def _remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for f in formulations:
            key = tuple(sorted([getattr(c, 'cid', 'N/A') for c in f['compounds']]))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def adaptive_evaluation(self, formulations: List[Dict], strategy: Dict, min_confidence: float, negotiation_round: int) -> Dict[str, Any]:
        approved, rejected = [], []
        adaptive_confidence = max(0.3, min_confidence * (1.0 - (negotiation_round * 0.15)))
        for f in formulations:
            score = self.calculate_overall_score(f, strategy, negotiation_round)
            f.update({'score': score, 'negotiation_round': negotiation_round, 'ai_decision': self.generate_ai_decision(f, score, adaptive_confidence)})
            (approved if score >= adaptive_confidence else rejected).append(f)
        return {'approved_formulations': approved, 'rejected_formulations': rejected, 'strategy': strategy}

    def calculate_overall_score(self, formulation, strategy, negotiation_round):
        prop_score = self.calculate_property_score(formulation, strategy)
        compat_score = self.calculate_compatibility_score(formulation.get('feasibility', {}))
        bonus = negotiation_round * 0.08
        return min(1.0, (prop_score * 0.7 + compat_score * 0.3) + bonus)
        
    def calculate_property_score(self, formulation: Dict, strategy: Dict) -> float:
        total_score, total_weight = 0, 0
        target_props = strategy.get('target_properties', {})
        if not isinstance(target_props, dict): return 0.5
        for prop, criteria in target_props.items():
            pred_val = formulation.get('predicted_properties', {}).get(prop, {}).get('value')
            if isinstance(pred_val, (int, float)):
                weight = criteria.get('weight', 0.1)
                total_score += self._score_property(pred_val, criteria) * weight
                total_weight += weight
        return total_score / max(total_weight, 0.1)

    def _score_property(self, pred_value, criteria):
        target, min_val, max_val = criteria.get('target'), criteria.get('min'), criteria.get('max')
        if target: return max(0, 1 - abs(pred_value - target) / target)
        if min_val: return 1.0 if pred_value >= min_val else pred_value / min_val
        if max_val: return 1.0 if pred_value <= max_val else max_val / pred_value
        return 0.5

    def calculate_compatibility_score(self, feasibility: Dict) -> float:
        return max(0, (1.0 - feasibility.get('risk_score', 0.5)) - (len(feasibility.get('compatibility_issues', [])) * 0.1))

    def generate_ai_decision(self, formulation, score, min_confidence):
        reasons = []
        if score >= 0.8: reasons.append("Excellent property match")
        elif score >= 0.6: reasons.append("Good performance")
        else: reasons.append("Acceptable trade-offs")
        reasons.append(f"{formulation.get('agent', 'Unknown')} strategy")
        return {'approved': score >= min_confidence, 'reasons': reasons}

    def clean_challenge_text(self, text: str) -> str:
        return ' '.join(text[:2000].split())

    def extract_json_from_response(self, response_text: str) -> str:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match: return match.group()
        raise ValueError("No JSON found in AI response")

