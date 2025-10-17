# ai_engine.py - Advanced AI with Gemini integration
import google.generativeai as genai
import json
import re
from typing import Dict, List, Any
import streamlit as st

class MaterialsAIEngine:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.property_databases = {
            'thermal': ['thermal_conductivity', 'specific_heat', 'viscosity', 'flash_point', 'pour_point'],
            'adsorption': ['surface_area', 'pore_volume', 'adsorption_capacity', 'selectivity', 'regeneration_energy'],
            'catalytic': ['turnover_frequency', 'selectivity', 'stability', 'overpotential', 'active_sites'],
            'mechanical': ['tensile_strength', 'youngs_modulus', 'hardness', 'toughness', 'elongation'],
            'optical': ['refractive_index', 'band_gap', 'transmittance', 'reflectance', 'absorption_coefficient']
        }
    
    def set_api_key(self, api_key: str):
        """Set Gemini API key and initialize model"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def interpret_challenge(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        """Use Gemini to interpret challenge and extract requirements"""
        
        prompt = f"""
        You are an expert materials scientist and chemical engineer. Analyze the following materials challenge and extract structured requirements.

        CHALLENGE: {challenge_text}
        MATERIAL TYPE: {material_type}

        Extract the following information as a JSON object:

        1. material_class: The primary material category
        2. critical_requirements: Must-have properties with minimum/maximum values
        3. performance_targets: Optimization goals with ideal ranges  
        4. safety_constraints: Toxicity, environmental, regulatory requirements
        5. practical_constraints: Cost, availability, synthesis complexity
        6. search_strategy: Recommended chemical classes and search terms
        7. priority_weights: Relative importance of each property (sum to 1.0)

        Return ONLY valid JSON, no other text.

        Example output format:
        {{
            "material_class": "coolant",
            "critical_requirements": {{
                "flash_point": {{"min": 150, "unit": "°C"}},
                "viscosity_100c": {{"max": 10, "unit": "cSt"}},
                "pfas_free": true
            }},
            "performance_targets": {{
                "thermal_conductivity": {{"min": 0.12, "target": 0.15, "unit": "W/m·K", "weight": 0.3}},
                "specific_heat": {{"min": 1800, "target": 2200, "unit": "J/kg·K", "weight": 0.2}}
            }},
            "safety_constraints": ["non_toxic", "non_flammable", "biodegradable"],
            "practical_constraints": ["cost_effective", "readily_available"],
            "search_strategy": {{
                "chemical_classes": ["silicones", "synthetic_esters", "polyalphaolefins"],
                "search_terms": ["siloxane", "polyol ester", "PAO", "mineral oil"]
            }},
            "priority_weights": {{
                "safety": 0.3,
                "performance": 0.4,
                "cost": 0.2,
                "availability": 0.1
            }}
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_str = self.extract_json_from_response(response.text)
            strategy = json.loads(json_str)
            
            # Merge into target_properties for compatibility
            target_properties = {}
            if 'critical_requirements' in strategy:
                target_properties.update(strategy['critical_requirements'])
            if 'performance_targets' in strategy:
                target_properties.update(strategy['performance_targets'])
            
            strategy['target_properties'] = target_properties
            return strategy
            
        except Exception as e:
            st.error(f"AI interpretation failed: {e}")
            # Return default strategy
            return self.get_default_strategy(material_type)
    
    def extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from Gemini response"""
        # Look for JSON pattern
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group()
        else:
            raise ValueError("No JSON found in AI response")
    
    def get_default_strategy(self, material_type: str) -> Dict[str, Any]:
        """Provide default strategy if AI fails"""
        default_strategies = {
            "Coolant/Lubricant": {
                "material_class": "coolant",
                "target_properties": {
                    "thermal_conductivity": {"min": 0.1, "target": 0.15, "unit": "W/m·K", "weight": 0.25},
                    "viscosity_100c": {"max": 10, "target": 5, "unit": "cSt", "weight": 0.2},
                    "flash_point": {"min": 150, "target": 200, "unit": "°C", "weight": 0.15},
                    "pour_point": {"max": -20, "target": -40, "unit": "°C", "weight": 0.1}
                },
                "search_strategy": {
                    "chemical_classes": ["silicones", "esters", "polyalphaolefins", "glycols"],
                    "search_terms": ["siloxane", "polyol ester", "PAO", "propylene glycol"]
                }
            },
            "Adsorbent": {
                "material_class": "adsorbent", 
                "target_properties": {
                    "surface_area": {"min": 500, "target": 1500, "unit": "m²/g", "weight": 0.3},
                    "pore_volume": {"min": 0.3, "target": 0.8, "unit": "cm³/g", "weight": 0.25},
                    "adsorption_capacity": {"min": 2, "target": 5, "unit": "mmol/g", "weight": 0.25}
                },
                "search_strategy": {
                    "chemical_classes": ["zeolites", "MOFs", "activated_carbons", "silicas"],
                    "search_terms": ["zeolite", "MOF", "activated carbon", "silica gel"]
                }
            }
        }
        
        return default_strategies.get(material_type, default_strategies["Coolant/Lubricant"])
    
    def generate_formulations(self, compounds_data: Dict, strategy: Dict) -> List[Dict]:
        """Generate optimized multi-compound formulations using AI"""
        
        formulations = []
        specialists = compounds_data.get('specialists', {})
        balanced = compounds_data.get('balanced', [])
        
        # Single compound formulations from balanced candidates
        for i, (compound, score) in enumerate(balanced[:5]):
            formulations.append({
                'compounds': [compound],
                'ratios': [1.0],
                'strategy': 'single_compound',
                'base_score': score
            })
        
        # Specialist-enhanced formulations
        if balanced and specialists:
            base_compound = balanced[0][0]  # Best balanced compound
            
            for prop_name, specialist_list in specialists.items():
                if specialist_list:
                    specialist = specialist_list[0]  # Best specialist for this property
                    
                    formulations.append({
                        'compounds': [base_compound, specialist],
                        'ratios': [0.7, 0.3],
                        'strategy': f'base_plus_{prop_name}_specialist',
                        'base_score': balanced[0][1] * 0.7 + 0.3  # Weighted score
                    })
        
        # Specialist combinations (no base)
        specialist_combinations = self.create_specialist_combinations(specialists)
        formulations.extend(specialist_combinations)
        
        return formulations
    
    def create_specialist_combinations(self, specialists: Dict) -> List[Dict]:
        """Create formulations combining multiple specialists"""
        combinations = []
        specialist_lists = list(specialists.values())
        
        # Take top specialist from each of first 3 properties
        top_specialists = []
        for prop_list in specialist_lists[:3]:
            if prop_list:
                top_specialists.append(prop_list[0])
        
        if len(top_specialists) >= 2:
            # Binary combination
            combinations.append({
                'compounds': top_specialists[:2],
                'ratios': [0.5, 0.5],
                'strategy': 'specialist_binary_combo',
                'base_score': 0.6
            })
        
        if len(top_specialists) >= 3:
            # Ternary combination  
            combinations.append({
                'compounds': top_specialists[:3],
                'ratios': [0.4, 0.3, 0.3],
                'strategy': 'specialist_ternary_combo', 
                'base_score': 0.65
            })
        
        return combinations
    
    def evaluate_and_rank_formulations(self, formulations: List[Dict], strategy: Dict, min_confidence: float) -> Dict[str, Any]:
        """Use AI to evaluate formulations and provide final ranking"""
        
        ranked_formulations = []
        
        for formulation in formulations:
            # Calculate property performance score
            property_score = self.calculate_property_score(formulation, strategy)
            
            # Calculate compatibility score
            feasibility = formulation.get('feasibility', {})
            compatibility_score = self.calculate_compatibility_score(feasibility)
            
            # Calculate practical score
            practical_score = self.calculate_practical_score(formulation)
            
            # Overall score (weighted average)
            overall_score = (
                property_score * 0.6 + 
                compatibility_score * 0.3 + 
                practical_score * 0.1
            )
            
            # AI decision
            ai_decision = self.generate_ai_decision(formulation, overall_score, min_confidence)
            
            # Update formulation with scores and decision
            formulation['score'] = overall_score
            formulation['ai_decision'] = ai_decision
            formulation['property_score'] = property_score
            formulation['compatibility_score'] = compatibility_score
            
            if ai_decision['approved'] or overall_score >= min_confidence:
                ranked_formulations.append(formulation)
        
        # Sort by score
        ranked_formulations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'top_formulations': ranked_formulations[:10],
            'search_metrics': {
                'compounds_evaluated': len(formulations) * 2,  # Estimate
                'formulations_generated': len(formulations),
                'formulations_approved': len([f for f in ranked_formulations if f['ai_decision']['approved']])
            },
            'strategy': strategy
        }
    
    def calculate_property_score(self, formulation: Dict, strategy: Dict) -> float:
        """Calculate how well formulation meets property targets"""
        total_score = 0
        total_weight = 0
        
        predicted_props = formulation.get('predicted_properties', {})
        target_props = strategy.get('target_properties', {})
        
        for prop_name, criteria in target_props.items():
            weight = criteria.get('weight', 0.1)
            target_value = criteria.get('target')
            min_value = criteria.get('min')
            max_value = criteria.get('max')
            
            pred_value = predicted_props.get(prop_name, {})
            if isinstance(pred_value, dict):
                pred_value = pred_value.get('value', 0)
            
            if target_value is not None and pred_value:
                # Score based on proximity to target
                normalized_diff = abs(pred_value - target_value) / max(target_value, 1)
                prop_score = max(0, 1 - normalized_diff)
            elif min_value is not None and pred_value:
                # Score based on exceeding minimum
                prop_score = 1.0 if pred_value >= min_value else pred_value / min_value
            elif max_value is not None and pred_value:
                # Score based on being below maximum  
                prop_score = 1.0 if pred_value <= max_value else max_value / pred_value
            else:
                prop_score = 0.5  # Default score
            
            total_score += prop_score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.1)
    
    def calculate_compatibility_score(self, feasibility: Dict) -> float:
        """Calculate compatibility score"""
        if not feasibility:
            return 0.7  # Default if no compatibility check
        
        risk_score = feasibility.get('risk_score', 0.5)
        issues = feasibility.get('compatibility_issues', [])
        
        base_score = 1.0 - risk_score
        # Penalize for each compatibility issue
        issue_penalty = len(issues) * 0.1
        
        return max(0, base_score - issue_penalty)
    
    def calculate_practical_score(self, formulation: Dict) -> float:
        """Calculate practical implementation score"""
        compounds = formulation.get('compounds', [])
        
        # Simple heuristic: fewer compounds = more practical
        compound_count = len(compounds)
        count_score = 1.0 if compound_count == 1 else 0.8 if compound_count == 2 else 0.6
        
        return count_score
    
    def generate_ai_decision(self, formulation: Dict, overall_score: float, min_confidence: float) -> Dict:
        """Generate AI decision with explanations"""
        
        approved = overall_score >= min_confidence
        
        # Generate reasons based on formulation characteristics
        reasons = []
        if overall_score >= 0.8:
            reasons.append("Excellent property match with target requirements")
        elif overall_score >= 0.6:
            reasons.append("Good overall performance with minor optimizations needed")
        else:
            reasons.append("Marginal performance - significant improvements required")
        
        # Add specific notes
        compounds = formulation.get('compounds', [])
        if len(compounds) == 1:
            reasons.append("Single compound formulation - simple implementation")
        else:
            reasons.append(f"Multi-compound formulation ({len(compounds)} components) - potential synergistic effects")
        
        feasibility = formulation.get('feasibility', {})
        if feasibility.get('risk_score', 0) < 0.3:
            reasons.append("Good chemical compatibility predicted")
        elif feasibility.get('risk_score', 0) < 0.6:
            reasons.append("Moderate compatibility risk - requires experimental validation")
        
        return {
            'approved': approved,
            'confidence': overall_score,
            'reasons': reasons,
            'recommendations': ["Proceed with experimental validation" if approved else "Consider alternative formulations"],
            'risk_factors': feasibility.get('compatibility_issues', [])
        }
