# app.py - COMPLETE PRODUCTION READY VERSION
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from ai_engine import MaterialsAIEngine
from pubchem_manager import PubChemManager
from mixture_predictor import MixturePredictor
from compatibility_checker import CompatibilityChecker

# Configure page
st.set_page_config(
    page_title="🧪 Advanced Materials Discovery AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .property-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #e3f2fd;
        border-radius: 15px;
        font-size: 0.85rem;
        border: 1px solid #bbdefb;
    }
    .approved-badge {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .rejected-badge {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedMaterialsApp:
    def __init__(self):
        self.ai_engine = MaterialsAIEngine()
        self.pubchem_manager = PubChemManager()
        self.mixture_predictor = MixturePredictor()
        self.compatibility_checker = CompatibilityChecker()
        
    def render_sidebar(self):
        """Render configuration sidebar"""
        with st.sidebar:
            st.header("⚙️ AI Configuration")
            
            # Gemini API Key input
            gemini_key = st.text_input(
                "🔑 Enter Gemini API Key:",
                type="password",
                help="Get your API key from: https://aistudio.google.com/app/apikey"
            )
            
            if gemini_key:
                st.session_state.gemini_key = gemini_key
                st.success("✅ API Key saved!")
            elif not hasattr(st.session_state, 'gemini_key'):
                st.warning("⚠️ Please enter Gemini API key to continue")
            
            st.markdown("---")
            
            # Search configuration
            st.subheader("🔍 Search Parameters")
            max_compounds = st.slider("Maximum compounds to analyze", 10, 200, 50)
            search_depth = st.selectbox(
                "Search depth", 
                ["Quick Scan", "Standard Analysis", "Comprehensive Search"],
                index=1
            )
            
            # Material type
            material_type = st.selectbox(
                "Material Category",
                [
                    "Coolant/Lubricant", "Adsorbent", "Catalyst", 
                    "Polymer", "Battery Material", "Pharmaceutical",
                    "Cosmetic", "Agricultural", "Custom"
                ]
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                min_confidence = st.slider("Minimum confidence threshold", 0.1, 1.0, 0.7)
                enable_compatibility = st.checkbox("Enable compatibility checking", True)
                enable_mixture_prediction = st.checkbox("Enable mixture prediction", True)
            
            return {
                'gemini_key': gemini_key,
                'max_compounds': max_compounds,
                'search_depth': search_depth,
                'material_type': material_type,
                'min_confidence': min_confidence,
                'enable_compatibility': enable_compatibility,
                'enable_mixture_prediction': enable_mixture_prediction
            }
    
    def render_main_interface(self):
        """Render main input interface"""
        st.markdown('<h1 class="main-header">🧪 Advanced Materials Discovery AI</h1>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Multi-Objective Chemical Formulation System</div>', unsafe_allow_html=True)
        
        # Two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Challenge Description")
            challenge_text = st.text_area(
                "Describe your materials challenge in detail:",
                height=200,
                placeholder="""Example: Need an immersion cooling fluid for data centers with:
• Flash point ≥ 150°C
• Kinematic viscosity ≤ 10 cSt at 100°C
• High thermal conductivity (> 0.12 W/m·K)
• Non-toxic, PFAS-free, environmentally friendly
• Chemically stable at 60°C for >1 year
• Compatible with electronics materials
• Cost-effective and readily available"""
            )
        
        with col2:
            st.subheader("🎯 Target Properties")
            st.info("""
            The AI will automatically extract and prioritize:
            - **Critical Requirements** (must-have)
            - **Performance Targets** (optimization goals)  
            - **Safety Constraints** (toxicity, environmental)
            - **Practical Considerations** (cost, availability)
            """)
            
            st.markdown("**Expected Output:**")
            st.write("• Top 10 optimized formulations")
            st.write("• Property predictions & confidence scores")
            st.write("• Compatibility analysis")
            st.write("• Implementation recommendations")
        
        return challenge_text
    
    def render_progress_tracker(self, phase, progress):
        """Render progress tracking"""
        phases = [
            "🔍 Interpreting challenge...",
            "🧪 Searching chemical space...", 
            "📊 Analyzing properties...",
            "🔄 Generating formulations...",
            "⚖️ Validating mixtures...",
            "🎯 Final ranking..."
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, phase_text in enumerate(phases[:phase]):
            status_text.text(phase_text)
            progress_bar.progress(progress)
            time.sleep(0.5)  # Simulate work
        
        return progress_bar, status_text
    
    def display_results(self, results, config):
        """Display comprehensive results"""
        st.success(f"✅ Analysis Complete! Found {len(results.get('top_formulations', []))} optimized formulations")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Compounds Analyzed", results.get('search_metrics', {}).get('compounds_evaluated', 0))
        with col2:
            st.metric("Formulations Generated", results.get('search_metrics', {}).get('formulations_generated', 0))
        with col3:
            st.metric("Approved Formulations", results.get('search_metrics', {}).get('formulations_approved', 0))
        with col4:
            avg_confidence = np.mean([f.get('ai_decision', {}).get('confidence', 0) 
                                    for f in results.get('top_formulations', [])])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Strategy overview
        with st.expander("📋 AI Strategy Overview", expanded=True):
            strategy = results.get('strategy', {})
            st.write(f"**Material Class:** {strategy.get('material_class', 'Custom')}")
            st.write("**Target Properties:**")
            for prop, criteria in strategy.get('target_properties', {}).items():
                st.write(f"- {prop}: {criteria}")
        
        # Top formulations
        st.subheader("🎯 Top Optimized Formulations")
        
        for i, formulation in enumerate(results.get('top_formulations', [])[:10]):
            self.display_formulation_card(i, formulation, config)
    
    def display_formulation_card(self, index, formulation, config):
        """Display individual formulation card"""
        with st.container():
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            
            # Header with score and status
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"Formulation #{index + 1}")
            
            with col2:
                score = formulation.get('score', 0)
                st.metric("Overall Score", f"{score:.2f}")
            
            with col3:
                decision = formulation.get('ai_decision', {})
                if decision.get('approved', False):
                    st.markdown(f'<div class="approved-badge">✅ APPROVED</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="rejected-badge">❌ REJECTED</div>', unsafe_allow_html=True)
            
            # Compounds and ratios
            st.write("**Composition:**")
            comp_cols = st.columns(len(formulation.get('compounds', [])))
            for idx, (compound, ratio) in enumerate(zip(formulation.get('compounds', []), 
                                                      formulation.get('ratios', []))):
                with comp_cols[idx]:
                    if hasattr(compound, 'iupac_name'):
                        st.write(f"**{compound.iupac_name}**")
                        st.write(f"CID: {compound.cid}")
                        st.write(f"Ratio: {ratio:.1%}")
                    else:
                        st.write(f"Compound {idx + 1}: {ratio:.1%}")
            
            # Predicted properties
            st.write("**Predicted Properties:**")
            props = formulation.get('predicted_properties', {})
            prop_text = ""
            for prop, value in props.items():
                if isinstance(value, dict):
                    prop_text += f"<span class='property-badge'>{prop}: {value.get('value', 'N/A')}</span>"
                else:
                    prop_text += f"<span class='property-badge'>{prop}: {value}</span>"
            st.markdown(prop_text, unsafe_allow_html=True)
            
            # AI Decision details
            decision = formulation.get('ai_decision', {})
            if decision.get('reasons'):
                st.write("**Decision Analysis:**")
                for reason in decision.get('reasons', []):
                    st.write(f"- {reason}")
            
            # Compatibility info
            feasibility = formulation.get('feasibility', {})
            if feasibility.get('compatibility_issues'):
                st.write("**⚠️ Compatibility Notes:**")
                for issue in feasibility.get('compatibility_issues', []):
                    st.write(f"- {issue}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def run_analysis(self, challenge_text, config):
        """Run complete analysis pipeline"""
        if not config['gemini_key']:
            st.error("❌ Please enter your Gemini API key in the sidebar")
            return None
        
        try:
            # Initialize AI engine with API key
            self.ai_engine.set_api_key(config['gemini_key'])
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: Challenge interpretation
            status_text.text("🔍 Phase 1: AI interpreting challenge...")
            strategy = self.ai_engine.interpret_challenge(challenge_text, config['material_type'])
            progress_bar.progress(20)
            
            # Phase 2: Compound discovery
            status_text.text("🧪 Phase 2: Exploring chemical space...")
            compounds_data = self.pubchem_manager.find_compounds(
                strategy, 
                config['max_compounds'],
                config['search_depth']
            )
            progress_bar.progress(40)
            
            # Phase 3: Formulation generation
            status_text.text("🔄 Phase 3: Generating optimized formulations...")
            formulations = self.ai_engine.generate_formulations(compounds_data, strategy)
            progress_bar.progress(60)
            
            # Phase 4: Mixture prediction
            if config['enable_mixture_prediction']:
                status_text.text("📊 Phase 4: Predicting mixture properties...")
                formulations = self.mixture_predictor.predict_all_properties(formulations, strategy)
                progress_bar.progress(75)
            
            # Phase 5: Compatibility checking
            if config['enable_compatibility']:
                status_text.text("⚖️ Phase 5: Validating chemical compatibility...")
                formulations = self.compatibility_checker.validate_all_formulations(formulations)
                progress_bar.progress(85)
            
            # Phase 6: AI evaluation and ranking
            status_text.text("🎯 Phase 6: AI evaluation and final ranking...")
            final_results = self.ai_engine.evaluate_and_rank_formulations(
                formulations, strategy, config['min_confidence']
            )
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            return final_results
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Try reducing the number of compounds or using Quick Scan mode")
            return None

def main():
    app = AdvancedMaterialsApp()
    
    # Render sidebar and get configuration
    config = app.render_sidebar()
    
    # Render main interface and get challenge
    challenge_text = app.render_main_interface()
    
    # Analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button(
            "🚀 START ADVANCED ANALYSIS", 
            type="primary", 
            use_container_width=True,
            disabled=not config['gemini_key']
        )
    
    # Run analysis when button clicked
    if analyze_btn and challenge_text:
        with st.spinner("🚀 Starting advanced materials analysis..."):
            results = app.run_analysis(challenge_text, config)
            
            if results:
                app.display_results(results, config)
    
    elif analyze_btn and not challenge_text:
        st.warning("⚠️ Please enter a challenge description first")
    
    # Examples section
    with st.expander("💡 Example Challenges", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Coolant", "Adsorbent", "Catalyst"])
        
        with tab1:
            st.code("""
Need immersion cooling fluid for AI data centers:
- Flash point ≥ 200°C, auto-ignition temperature > 300°C
- Kinematic viscosity: 3-8 cSt at 100°C, < 30 cSt at 40°C  
- Thermal conductivity > 0.13 W/m·K, specific heat > 2000 J/kg·K
- Dielectric constant > 2.2, dielectric strength > 35 kV/2.5mm
- PFAS-free, non-toxic, biodegradable
- Material compatible with copper, aluminum, plastics
- Long-term stability at 60-80°C operating temperature
- Cost < $50/kg, readily available globally
            """)
        
        with tab2:
            st.code("""
Need porous adsorbent for CO2 capture from flue gas:
- Surface area > 1500 m²/g, pore volume > 0.8 cm³/g
- CO2 adsorption capacity > 3 mmol/g at 25°C, 1 bar
- CO2/N2 selectivity > 50, fast adsorption kinetics
- Stable in humid conditions (10-90% RH)
- Low regeneration energy < 2 MJ/kg CO2
- Mechanical strength for pelletization
- Stable for > 10,000 adsorption-desorption cycles
- Raw material cost < $20/kg
            """)
        
        with tab3:
            st.code("""
Need heterogeneous catalyst for hydrogen production:
- High activity for water splitting reaction
- Turnover frequency > 10 s⁻¹ at 1.23 V vs RHE
- Overpotential < 200 mV at 10 mA/cm²
- Stability > 1000 hours in acidic/alkaline conditions
- Earth-abundant elements (no precious metals)
- Easy synthesis and scale-up
- Cost < $100/kg catalyst
- High selectivity (>99%) for hydrogen production
            """)

if __name__ == "__main__":
    main()
