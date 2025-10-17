# ai_engine.py - CORRECTED INDENTATION
import google.generativeai as genai
import json
import re
from typing import Dict, List, Any
import streamlit as st
import time

class MaterialsAIEngine:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.property_databases = {
            'thermal': ['thermal_conductivity', 'specific_heat', 'viscosity', 'flash_point', 'pour_point'],
            'adsorption': ['surface_area', 'pore_volume', 'adsorption_capacity', 'selectivity'],
            'catalytic': ['turnover_frequency', 'selectivity', 'stability'],
            'mechanical': ['tensile_strength', 'youngs_modulus', 'hardness'],
            'optical': ['refractive_index', 'band_gap', 'transmittance']
        }
    
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

    def interpret_challenge(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        if not self.model:
            st.warning("⚠️ Using default strategy - AI model not available")
            return self.get_default_strategy(material_type)
        
        clean_text = self.clean_challenge_text(challenge_text)
        
        prompt = f"""
        You are an expert materials scientist. Analyze this challenge and extract key requirements.

        CHALLENGE: {clean_text}
        MATERIAL TYPE: {material_type}

        Extract the key technical requirements and return as JSON with:
        - material_class
        - target_properties (with min/max/target/weight)
        - safety_constraints
        - search_strategy with chemical_classes and search_terms

        Example:
        {{
            "material_class": "coolant",
            "target_properties": {{
                "thermal_conductivity": {{"min": 0.1, "target": 0.15, "unit": "W/m·K", "weight": 0.3}},
                "viscosity_100c": {{"max": 10, "target": 5, "unit": "cSt", "weight": 0.2}},
                "flash_point": {{"min": 150, "target": 200, "unit": "°C", "weight": 0.15}}
            }},
            "safety_constraints": ["non_toxic", "pfas_free", "non_flammable"],
            "search_strategy": {{
                "chemical_classes": ["silicones", "synthetic_esters", "polyalphaolefins"],
                "search_terms": ["siloxane", "polyol ester", "PAO", "mineral oil"]
            }}
        }}

        Return ONLY valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_str = self.extract_json_from_response(response.text)
            strategy = json.loads(json_str)
            strategy = self.validate_strategy(strategy, material_type)
            return strategy
            
        except Exception as e:
            st.warning(f"⚠️ AI interpretation failed, using default strategy: {str(e)}")
            return self.get_default_strategy(material_type)

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

    def validate_strategy(self, strategy: Dict, material_type: str) -> Dict:
        if 'material_class' not in strategy:
            strategy['material_class'] = material_type.lower()
        
        if 'target_properties' not in strategy:
            strategy['target_properties'] = self.get_default_strategy(material_type)['target_properties']
        
        if 'search_strategy' not in strategy:
            strategy['search_strategy'] = self.get_default_strategy(material_type)['search_strategy']
        
        if 'safety_constraints' not in strategy:
            strategy['safety_constraints'] = ['non_toxic', 'pfas_free']
        
        return strategy

    def get_default_strategy(self, material_type: str) -> Dict[str, Any]:
        default_strategies = {
            "Coolant/Lubricant": {
                "material_class": "coolant",
                "target_properties": {
                    "thermal_conductivity": {"min": 0.1, "target": 0.15, "unit": "W/m·K", "weight": 0.25},
                    "viscosity_100c": {"max": 10, "target": 5, "unit": "cSt", "weight": 0.2},
                    "flash_point": {"min": 150, "target": 200, "unit": "°C", "weight": 0.15},
                    "pour_point": {"max": -20, "target": -40, "unit": "°C", "weight": 0.1},
                    "specific_heat": {"min": 1800, "target": 2200, "unit": "J/kg·K", "weight": 0.15}
                },
                "safety_constraints": ["non_toxic", "pfas_free", "non_flammable"],
                "search_strategy": {
                    "chemical_classes": ["silicones", "esters", "polyalphaolefins", "glycols"],
                    "search_terms": [
                        "dimethyl siloxane", "polydimethylsiloxane", "hexamethyldisiloxane",
                        "mineral oil", "paraffin oil", "polyalphaolefin", "propylene glycol",
                        "ethylene glycol", "ester oil", "polyol ester"
                    ]
                }
            },
            "Adsorbent": {
                "material_class": "adsorbent", 
                "target_properties": {
                    "surface_area": {"min": 500, "target": 1500, "unit": "m²/g", "weight": 0.4},
                    "pore_volume": {"min": 0.3, "target": 0.8, "unit": "cm³/g", "weight": 0.3},
                    "adsorption_capacity": {"min": 2, "target": 5, "unit": "mmol/g", "weight": 0.3}
                },
                "safety_constraints": ["non_toxic", "stable"],
                "search_strategy": {
                    "chemical_classes": ["zeolites", "activated_carbons", "silicas", "MOFs"],
                    "search_terms": ["zeolite", "activated carbon", "silica gel", "alumina"]
                }
            }
        }
        
        return default_strategies.get(material_type, default_strategies["Coolant/Lubricant"])

    def generate_formulations(self, compounds_data: Dict, strategy: Dict) -> List[Dict]:
        formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        
        st.write(f"📊 Generating formulations from {len(balanced)} balanced compounds and {len(specialists)} specialist categories...")
        
        for i, (compound, score) in enumerate(balanced[:3]):
            formulations.append({
                'compounds': [compound],
                'ratios': [1.0],
                'strategy': f'single_compound_{i+1}',
                'base_score': score,
                'composition_type': 'single'
            })
        
        if balanced and specialists:
            base_compound, base_score = balanced[0]
            
            for prop_name, specialist_list in specialists.items():
                if specialist_list and len(specialist_list) > 0:
                    specialist = specialist_list[0]
                    
                    formulations.append({
                        'compounds': [base_compound, specialist],
                        'ratios': [0.7, 0.3],
                        'strategy': f'base_plus_{prop_name}_specialist',
                        'base_score': base_score * 0.7 + 0.3,
                        'composition_type': 'binary'
                    })
        
        if len(balanced) >= 2:
            for i in range(min(2, len(balanced))):
                for j in range(i+1, min(3, len(balanced))):
                    comp1, score1 = balanced[i]
                    comp2, score2 = balanced[j]
                    
                    formulations.append({
                        'compounds': [comp1, comp2],
                        'ratios': [0.5, 0.5],
                        'strategy': f'balanced_combo_{i+1}_{j+1}',
                        'base_score': (score1 + score2) / 2,
                        'composition_type': 'binary'
                    })
        
        st.write(f"✅ Generated {len(formulations)} formulations")
        return formulations

    def evaluate_and_rank_formulations(self, formulations: List[Dict], strategy: Dict, min_confidence: float) -> Dict[str, Any]:
        if not formulations:
            st.warning("❌ No formulations to evaluate")
            return {
                'top_formulations': [],
                'search_metrics': {'compounds_evaluated': 0, 'formulations_generated': 0, 'formulations_approved': 0},
                'strategy': strategy
            }
        
        ranked_formulations = []
        
        for formulation in formulations:
            property_score = self.calculate_property_score(formulation, strategy)
            feasibility = formulation.get('feasibility', {})
            compatibility_score = self.calculate_compatibility_score(feasibility)
            overall_score = (property_score * 0.7 + compatibility_score * 0.3)
            ai_decision = self.generate_ai_decision(formulation, overall_score, min_confidence)
            
            formulation['score'] = overall_score
            formulation['ai_decision'] = ai_decision
            formulation['property_score'] = property_score
            formulation['compatibility_score'] = compatibility_score
            
            if ai_decision['approved']:
                ranked_formulations.append(formulation)
        
        ranked_formulations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'top_formulations': ranked_formulations[:10],
            'search_metrics': {
                'compounds_evaluated': len(formulations) * 2,
                'formulations_generated': len(formulations),
                'formulations_approved': len(ranked_formulations)
            },
            'strategy': strategy
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

    def generate_ai_decision(self, formulation: Dict, overall_score: float, min_confidence: float) -> Dict:
        approved = overall_score >= min_confidence
        
        reasons = []
        if overall_score >= 0.8:
            reasons.append("Excellent property match with requirements")
        elif overall_score >= 0.6:
            reasons.append("Good overall performance")
        else:
            reasons.append("Marginal performance - needs improvement")
        
        compounds = formulation.get('compounds', [])
        if len(compounds) == 1:
            reasons.append("Single compound - simple implementation")
        else:
            reasons.append(f"Multi-compound formulation - potential synergies")
        
        feasibility = formulation.get('feasibility', {})
        if feasibility.get('risk_score', 0) < 0.3:
            reasons.append("Good chemical compatibility")
        
        return {
            'approved': approved,
            'confidence': overall_score,
            'reasons': reasons,
            'recommendations': ["Proceed with experimental validation" if approved else "Consider alternative formulations"],
            'risk_factors': feasibility.get('compatibility_issues', [])
        }
