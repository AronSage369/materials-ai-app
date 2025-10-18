import streamlit as st
import json
import time
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ai_strategist import AIStrategist
from creative_ai_engine import CreativeAIEngine
from computational_predictor import ComputationalPredictor
from pubchem_manager import PubChemManager
from property_predictor import AdvancedPropertyPredictor
from compatibility_checker import CompatibilityChecker

class AdvancedMaterialsDiscoveryApp:
    def __init__(self):
        st.set_page_config(
            page_title="Advanced Materials Discovery AI",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.ai_strategist = None
            st.session_state.creative_engine = None
            st.session_state.computational_predictor = None
            st.session_state.pubchem_manager = None
            st.session_state.property_predictor = None
            st.session_state.compatibility_checker = None

    def run_advanced_analysis(self, challenge_text: str, config: Dict) -> Dict[str, Any]:
        """Run advanced AI-driven analysis"""
        # Step 1: AI Strategic Thinking
        with st.spinner("🧠 AI is thinking deeply about your challenge..."):
            strategy = st.session_state.ai_strategist.think_about_challenge(
                challenge_text, config['material_type']
            )
        
        # Step 2: Intelligent PubChem Search
        with st.spinner("🔍 AI is searching for relevant compounds..."):
            search_terms = (strategy.get('search_strategy', {})
                          .get('primary_terms', []) + 
                          strategy.get('search_strategy', {})
                          .get('innovative_terms', []))
            
            compounds = st.session_state.pubchem_manager.find_compounds_ai_driven(
                strategy, config['material_type'], search_terms
            )
        
        # Step 3: Creative Formulation Generation
        with st.spinner("🤖 Creative AI agents are designing formulations..."):
            formulations = st.session_state.creative_engine.creative_formulation_generation(
                compounds, strategy, config['innovation_factor']
            )
        
        # Step 4: Advanced Property Prediction
        with st.spinner("📊 Running computational predictions..."):
            target_props = self._extract_target_properties(strategy)
            formulations = st.session_state.property_predictor.predict_all_properties(
                formulations, target_props
            )
            
            # Add computational predictions
            for formulation in formulations:
                comp_predictions = st.session_state.computational_predictor.predict_advanced_properties(
                    formulation, target_props
                )
                formulation['computational_predictions'] = comp_predictions
        
        # Step 5: Compatibility Analysis
        with st.spinner("⚠️ Analyzing chemical compatibility..."):
            formulations = st.session_state.compatibility_checker.validate_all_formulations(formulations)
        
        # Step 6: AI Evaluation and Ranking
        with st.spinner("🎯 AI is evaluating and ranking formulations..."):
            approved_formulations = self._ai_evaluation(formulations, strategy, config)
        
        return {
            'strategy': strategy,
            'approved_formulations': approved_formulations,
            'total_compounds_found': len(compounds),
            'total_formulations_generated': len(formulations),
            'ai_thinking': strategy.get('scientific_analysis', ''),
            'innovative_ideas': strategy.get('innovative_ideas', [])
        }

    def _extract_target_properties(self, strategy: Dict) -> Dict[str, Any]:
        """Extract target properties from AI strategy"""
        # Convert AI strategy to property targets
        target_props = {}
        
        # Add computational properties based on challenge type
        scientific_analysis = strategy.get('scientific_analysis', '').lower()
        
        if any(word in scientific_analysis for word in ['solar', 'photo', 'light']):
            target_props.update({
                'band_gap': {'target': 1.8, 'min': 1.2, 'max': 3.0, 'importance': 'high'},
                'absorption_spectrum': {'target': 80, 'min': 50, 'importance': 'high'},
                'quantum_efficiency': {'target': 70, 'min': 30, 'importance': 'high'}
            })
        
        if any(word in scientific_analysis for word in ['electronic', 'conductor', 'mobility']):
            target_props.update({
                'electron_mobility': {'target': 10, 'min': 1, 'importance': 'high'},
                'hole_mobility': {'target': 10, 'min': 1, 'importance': 'high'}
            })
        
        return target_props

    def _ai_evaluation(self, formulations: List[Dict], strategy: Dict, config: Dict) -> List[Dict]:
        """AI-driven evaluation of formulations"""
        approved = []
        
        for formulation in formulations:
            # Multi-factor scoring
            innovation_score = formulation.get('innovation_score', 0.5)
            strategy_score = formulation.get('strategy_alignment', 0.5)
            computational_score = self._calculate_computational_score(formulation)
            compatibility_score = 1 - formulation.get('compatibility_risk', 0)
            
            # Weighted overall score
            overall_score = (innovation_score * 0.3 + 
                           strategy_score * 0.25 + 
                           computational_score * 0.25 + 
                           compatibility_score * 0.2)
            
            formulation['overall_score'] = overall_score
            
            if overall_score >= config['min_approval_score']:
                approved.append(formulation)
        
        # Sort by overall score
        approved.sort(key=lambda x: x['overall_score'], reverse=True)
        return approved

    def _calculate_computational_score(self, formulation: Dict) -> float:
        """Calculate score based on computational predictions"""
        comp_predictions = formulation.get('computational_predictions', {})
        if not comp_predictions:
            return 0.5
        
        # Average confidence of computational predictions
        confidences = [pred.get('confidence', 0) for pred in comp_predictions.values()]
        return np.mean(confidences) if confidences else 0.5



# Update the main function in app.py to use the new system
