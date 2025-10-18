# app.py - COMPLETE FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Any  # ADD THIS IMPORT
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
    .negotiation-badge {
        background-color: #fff3e0;
        color: #ef6c00;
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
                max_negotiation_rounds = st.slider("Max negotiation rounds", 1, 10, 5)
                enable_compatibility = st.checkbox("Enable compatibility checking", True)
                enable_mixture_prediction = st.checkbox("Enable mixture prediction", True)
                enable_mole_calculations = st.checkbox("Enable mole percentage calculations", True)
            
            return {
                'gemini_key': gemini_key,
                'max_compounds': max_compounds,
                'search_depth': search_depth,
                'material_type': material_type,
                'min_confidence': min_confidence,
                'max_negotiation_rounds': max_negotiation_rounds,
                'enable_compatibility': enable_compatibility,
                'enable_mixture_prediction': enable_mixture_prediction,
                'enable_mole_calculations': enable_mole_calculations
            }
    
    def render_main_interface(self):
        """Render main input interface"""
        st.markdown('<h1 class="main-header">🧪 Advanced Materials Discovery AI</h1>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Multi-Agent Adaptive Formulation System</div>', unsafe_allow_html=True)
        
        # Two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Challenge Description")
            challenge_text = st.text_area(
                "Describe your materials challenge in detail:",
                height=200,
                placeholder="""Example: Need an immersion cooling fluid for data centers with:
• Flash point ≥ 200°C, auto-ignition temperature > 300°C
• Kinematic viscosity: 3-8 cSt at 100°C, < 30 cSt at 40°C  
• Thermal conductivity > 0.13 W/m·K, specific heat > 2000 J/kg·K
• Dielectric constant > 2.2, dielectric strength > 35 kV/2.5mm
• PFAS-free, non-toxic, biodegradable
• Material compatible with copper, aluminum, plastics
• Long-term stability at 60-80°C operating temperature
• Cost < $50/kg, readily available globally"""
            )
        
        with col2:
            st.subheader("🎯 Advanced Features")
            st.info("""
            **New AI Capabilities:**
            - Multi-Agent System (4 specialized agents)
            - Adaptive Negotiation (progressive relaxation)
            - Mole Percentage Calculations
            - Complex Formulations (3-5 compounds)
            - Both Approved & Rejected Results
            - Intelligent Fallbacks
            """)
            
            st.markdown("**Expected Output:**")
            st.write("• Multi-round negotiation results")
            st.write("• Complex formulations (3-5 compounds)")
            st.write("• Mole percentage compositions")
            st.write("• Agent-specific strategies")
            st.write("• Comprehensive analysis")
        
        return challenge_text
    
    def run_analysis(self, challenge_text, config):
        """Run adaptive multi-round analysis with negotiation"""
        if not config['gemini_key']:
            st.error("❌ Please enter your Gemini API key in the sidebar")
            return None
        
        try:
            self.ai_engine.set_api_key(config['gemini_key'])
            
            all_results = []
            max_rounds = config['max_negotiation_rounds']
            
            for round_num in range(max_rounds):
                st.write(f"---")
                st.subheader(f"🔄 Negotiation Round {round_num + 1}")
                
                # Phase 1: Adaptive challenge interpretation
                with st.spinner(f"Round {round_num + 1}: AI interpreting challenge..."):
                    strategy = self.ai_engine.interpret_challenge(challenge_text, config['material_type'], round_num)
                
                # Phase 2: Compound discovery
                with st.spinner(f"Round {round_num + 1}: Exploring chemical space..."):
                    compounds_data = self.pubchem_manager.find_compounds(
                        strategy, 
                        config['max_compounds'],
                        config['search_depth']
                    )
                
                # Phase 3: Multi-agent formulation generation
                with st.spinner(f"Round {round_num + 1}: Multi-agent formulation generation..."):
                    formulations = self.ai_engine.multi_agent_formulation_generation(compounds_data, strategy)
                
                # Phase 4: Calculate mole percentages if enabled
                if config['enable_mole_calculations'] and formulations:
                    with st.spinner(f"Round {round_num + 1}: Calculating mole percentages..."):
                        formulations = self.ai_engine.calculate_mole_percentages(formulations)
                
                # Phase 5: Mixture prediction
                if config['enable_mixture_prediction'] and formulations:
                    with st.spinner(f"Round {round_num + 1}: Predicting mixture properties..."):
                        formulations = self.mixture_predictor.predict_all_properties(formulations, strategy)
                
                # Phase 6: Compatibility checking
                if config['enable_compatibility'] and formulations:
                    with st.spinner(f"Round {round_num + 1}: Validating chemical compatibility..."):
                        formulations = self.compatibility_checker.validate_all_formulations(formulations)
                
                # Phase 7: Adaptive evaluation
                with st.spinner(f"Round {round_num + 1}: AI evaluation and ranking..."):
                    round_results = self.ai_engine.adaptive_evaluation(
                        formulations, strategy, config['min_confidence'], round_num
                    )
                
                all_results.append(round_results)
                
                # Check if we have enough approved formulations
                approved_count = len(round_results.get('approved_formulations', []))
                if approved_count >= 5 or round_num == max_rounds - 1:
                    st.success(f"✅ Completed {round_num + 1} negotiation rounds with {approved_count} approved formulations")
                    break
                else:
                    st.warning(f"⚠️ Only {approved_count} approved formulations. Starting next negotiation round...")
            
            # Combine results from all rounds
            final_results = self.combine_negotiation_results(all_results)
            return final_results
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Try reducing the number of compounds or using Quick Scan mode")
            return None

   def combine_negotiation_results(self, all_results: List[Dict]) -> Dict[str, Any]:
    """Combine results from all negotiation rounds with improved filtering"""
    all_approved = []
    all_rejected = []
    total_metrics = {
        'compounds_evaluated': 0,
        'formulations_generated': 0,
        'formulations_approved': 0,
        'negotiation_rounds': len(all_results)
    }
    
    for result in all_results:
        all_approved.extend(result.get('approved_formulations', []))
        all_rejected.extend(result.get('rejected_formulations', []))
        
        metrics = result.get('search_metrics', {})
        total_metrics['compounds_evaluated'] += metrics.get('compounds_evaluated', 0)
        total_metrics['formulations_generated'] += metrics.get('formulations_generated', 0)
        total_metrics['formulations_approved'] += metrics.get('formulations_approved', 0)
    
    # Remove duplicates and sort by score
    all_approved = self.remove_duplicate_formulations(all_approved)
    all_rejected = self.remove_duplicate_formulations(all_rejected)
    
    all_approved.sort(key=lambda x: x.get('score', 0), reverse=True)
    all_rejected.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # ENHANCED: Show up to 20 approved formulations
    # If more than 20 approved, filter by confidence threshold
    if len(all_approved) > 20:
        # Keep top 20 by score (which already considers confidence)
        all_approved = all_approved[:20]
        st.info(f"📊 Showing top 20 approved formulations from {len(all_approved)} total (filtered by confidence)")
    
    # ENHANCED: Show more rejected for analysis (up to 15)
    top_rejected = all_rejected[:15]
    
    return {
        'approved_formulations': all_approved,
        'rejected_formulations': top_rejected,
        'search_metrics': total_metrics,
        'strategy': all_results[-1].get('strategy', {}) if all_results else {},
        'negotiation_summary': f"Completed {len(all_results)} rounds with {len(all_approved)} approved formulations"
    }

    def remove_duplicate_formulations(self, formulations: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations"""
        seen = set()
        unique = []
        
        for formulation in formulations:
            if 'compounds' in formulation:
                cids = tuple(sorted([getattr(c, 'cid', 'unknown') for c in formulation['compounds']]))
                if cids not in seen:
                    seen.add(cids)
                    unique.append(formulation)
        
        return unique

    def display_results(self, results, config):
        """Display comprehensive results with both approved and rejected formulations"""
        
        st.success(f"✅ {results.get('negotiation_summary', 'Analysis complete!')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = results.get('search_metrics', {})
        with col1:
            st.metric("Compounds Analyzed", metrics.get('compounds_evaluated', 0))
        with col2:
            st.metric("Formulations Generated", metrics.get('formulations_generated', 0))
        with col3:
            st.metric("Approved Formulations", metrics.get('formulations_approved', 0))
        with col4:
            st.metric("Negotiation Rounds", metrics.get('negotiation_rounds', 1))
        
        # Strategy overview
        with st.expander("📋 Final AI Strategy Overview", expanded=True):
            strategy = results.get('strategy', {})
            st.write(f"**Material Class:** {strategy.get('material_class', 'Custom')}")
            st.write("**Final Target Properties:**")
            for prop, criteria in strategy.get('target_properties', {}).items():
                st.write(f"- {prop}: {criteria}")
        
        # Approved formulations
        approved = results.get('approved_formulations', [])
        if approved:
            st.subheader("✅ Approved Formulations")
            st.info("These formulations met all criteria and are recommended for further development.")
            
            for i, formulation in enumerate(approved):
                self.display_formulation_card(i, formulation, config, approved=True)
        else:
            st.warning("❌ No formulations met the approval criteria")
        
        # Show rejected formulations with explanations
        rejected = results.get('rejected_formulations', [])
        if rejected:
            st.subheader("⚠️ Promising Rejected Formulations")
            st.info("""
            These formulations showed promise but didn't meet all criteria. 
            They may be valuable for further investigation with relaxed requirements.
            """)
            
            for i, formulation in enumerate(rejected):
                self.display_formulation_card(i, formulation, config, approved=False)
    
    def display_formulation_card(self, index, formulation, config, approved=True):
        """Display formulation card with approval status and mole percentages"""
        
        status_color = "✅" if approved else "⚠️"
        status_text = "APPROVED" if approved else "REJECTED"
        status_class = "approved-badge" if approved else "rejected-badge"
        
        with st.container():
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            
            # Header with status and agent info
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"{status_color} Formulation #{index + 1}")
                agent = formulation.get('agent', 'unknown').capitalize()
                negotiation_round = formulation.get('negotiation_round', 0) + 1
                st.write(f"**Agent:** {agent} | **Round:** {negotiation_round} | **Risk:** {formulation.get('risk_level', 'medium').capitalize()}")
            
            with col2:
                score = formulation.get('score', 0)
                st.metric("Overall Score", f"{score:.2f}")
            
            with col3:
                st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
            
            # Compounds and ratios
            compounds = formulation.get('compounds', [])
            ratios = formulation.get('ratios', [])
            mole_ratios = formulation.get('mole_ratios', [])
            
            st.write("**Composition:**")
            
            # Create columns for compounds
            comp_cols = st.columns(len(compounds))
            
            for idx, compound in enumerate(compounds):
                with comp_cols[idx]:
                    # Basic compound info
                    if hasattr(compound, 'iupac_name'):
                        comp_name = compound.iupac_name
                        # Shorten long names
                        if len(comp_name) > 30:
                            comp_name = comp_name[:27] + "..."
                        st.write(f"**{comp_name}**")
                    else:
                        st.write(f"**Compound {idx + 1}**")
                    
                    if hasattr(compound, 'cid'):
                        st.write(f"CID: {compound.cid}")
                    
                    if hasattr(compound, 'molecular_formula'):
                        st.write(f"Formula: {compound.molecular_formula}")
                    
                    if hasattr(compound, 'molecular_weight'):
                        st.write(f"MW: {compound.molecular_weight:.1f} g/mol")
                    
                    # Mass percentage
                    if idx < len(ratios):
                        mass_pct = ratios[idx] * 100
                        st.write(f"**Mass: {mass_pct:.1f}%**")
                    
                    # Mole percentage if available
                    if config['enable_mole_calculations'] and idx < len(mole_ratios):
                        mole_pct = mole_ratios[idx] * 100
                        st.write(f"**Moles: {mole_pct:.1f}%**")
            
            # Predicted properties
            st.write("**Predicted Properties:**")
            props = formulation.get('predicted_properties', {})
            prop_text = ""
            for prop, value in props.items():
                if isinstance(value, dict):
                    val = value.get('value', 'N/A')
                    unit = value.get('unit', '')
                    prop_text += f"<span class='property-badge'>{prop}: {val} {unit}</span>"
                else:
                    prop_text += f"<span class='property-badge'>{prop}: {value}</span>"
            st.markdown(prop_text, unsafe_allow_html=True)
            
            # AI Decision details
            decision = formulation.get('ai_decision', {})
            if decision.get('reasons'):
                st.write("**AI Analysis:**")
                for reason in decision.get('reasons', []):
                    st.write(f"- {reason}")
            
            # Compatibility info
            feasibility = formulation.get('feasibility', {})
            if feasibility.get('compatibility_issues'):
                st.write("**⚠️ Compatibility Notes:**")
                for issue in feasibility.get('compatibility_issues', []):
                    st.write(f"- {issue}")
            
            st.markdown('</div>', unsafe_allow_html=True)

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
            "🚀 START ADVANCED MULTI-AGENT ANALYSIS", 
            type="primary", 
            use_container_width=True,
            disabled=not config['gemini_key']
        )
    
    # Run analysis when button clicked
    if analyze_btn and challenge_text:
        with st.spinner("🚀 Starting advanced multi-agent materials analysis..."):
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

