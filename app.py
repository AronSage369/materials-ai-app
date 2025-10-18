# app.py - REWRITTEN WITH ENHANCED UI AND DETAILED RESULT VISUALIZATION
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from ai_engine import MaterialsAIEngine
from pubchem_manager import PubChemManager
from mixture_predictor import MixturePredictor
from compatibility_checker import CompatibilityChecker
from property_predictor import AdvancedPropertyPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🧪 Advanced Materials Discovery AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR A POLISHED LOOK ---
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; font-weight: 700; }
    .sub-header { font-size: 1.4rem; color: #2e86ab; margin-bottom: 2rem; text-align: center; }
    .result-card { 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 6px solid #1f77b4; 
        background-color: #ffffff; 
        margin-bottom: 1.5rem; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    .result-card:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
    .approved-badge { background-color: #e8f5e9; color: #2e7d32; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; border: 1px solid #a5d6a7; }
    .rejected-badge { background-color: #ffebee; color: #c62828; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; border: 1px solid #ef9a9a; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

class AdvancedMaterialsApp:
    def __init__(self):
        # Initialize all modules in session state for persistence
        if 'ai_engine' not in st.session_state:
            st.session_state.ai_engine = MaterialsAIEngine()
            st.session_state.pubchem_manager = PubChemManager()
            st.session_state.mixture_predictor = MixturePredictor()
            st.session_state.compatibility_checker = CompatibilityChecker()
            st.session_state.property_predictor = AdvancedPropertyPredictor()
    
    # Use properties for easy access to modules
    @property
    def ai_engine(self): return st.session_state.ai_engine
    @property
    def pubchem_manager(self): return st.session_state.pubchem_manager
    @property
    def mixture_predictor(self): return st.session_state.mixture_predictor
    @property
    def compatibility_checker(self): return st.session_state.compatibility_checker
    @property
    def property_predictor(self): return st.session_state.property_predictor
        
    def render_sidebar(self) -> Dict:
        """Renders the sidebar for configuration and returns the config dictionary."""
        with st.sidebar:
            st.header("⚙️ AI Configuration")
            gemini_key = st.text_input("🔑 Enter Gemini API Key:", type="password", help="Get your API key from Google AI Studio")
            if gemini_key: 
                st.session_state.gemini_key = gemini_key
                self.ai_engine.set_api_key(gemini_key)

            st.markdown("---")
            st.subheader("🔍 Search Parameters")
            config = {
                'max_compounds': st.slider("Max Compounds to Analyze", 10, 200, 50, 10),
                'search_depth': st.select_slider("Search Depth", ["Quick", "Standard", "Comprehensive"], value="Standard"),
                'material_type': st.selectbox("Material Category", ["Coolant/Lubricant", "Adsorbent"]),
                'gemini_key': st.session_state.get('gemini_key')
            }
            
            with st.expander("🔬 Advanced Options", expanded=True):
                config.update({
                    'min_confidence': st.slider("Min Approval Score", 0.1, 1.0, 0.5, 0.05),
                    'max_negotiation_rounds': st.slider("Max Negotiation Rounds", 1, 10, 3),
                    'max_approved_formulations': st.slider("Max Formulations to Show", 5, 30, 15)
                })
            return config
    
    def render_main_interface(self) -> str:
        """Renders the main part of the UI for user input."""
        st.markdown('<h1 class="main-header">🧪 Advanced Materials Discovery AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Multi-Agent Adaptive Formulation System</p>', unsafe_allow_html=True)
        
        challenge_text = st.text_area("📝 **Challenge Description**", height=150, 
                                      placeholder="Example: Design a non-toxic, biodegradable coolant for data centers with a thermal conductivity > 0.15 W/mK and a flash point > 200°C...")
        return challenge_text
    
    def run_full_analysis(self, challenge_text: str, config: Dict) -> Dict:
        """Orchestrates the entire multi-round analysis pipeline."""
        final_results = {'approved_formulations': [], 'rejected_formulations': [], 'search_metrics': {}, 'strategy': {}}
        all_approved_this_run = []

        with st.status("🚀 Launching Multi-Agent Analysis...", expanded=True) as status:
            for round_num in range(config['max_negotiation_rounds']):
                status.write(f"**🔄 Starting Negotiation Round {round_num + 1} of {config['max_negotiation_rounds']}...**")
                
                strategy = self.ai_engine.interpret_challenge(challenge_text, config['material_type'], round_num)
                compounds = self.pubchem_manager.find_compounds(strategy, config['max_compounds'], config['search_depth'])
                formulations = self.ai_engine.multi_agent_formulation_generation(compounds, strategy)
                
                formulations = self.ai_engine.calculate_mole_percentages(formulations)
                formulations = self.property_predictor.predict_all_properties(formulations, strategy)
                formulations = self.compatibility_checker.validate_all_formulations(formulations)
                
                round_results = self.ai_engine.adaptive_evaluation(formulations, strategy, config['min_confidence'], round_num)
                
                approved_this_round = round_results.get('approved_formulations', [])
                all_approved_this_run.extend(approved_this_round)
                
                status.write(f"✅ Round {round_num + 1} complete. Found {len(approved_this_round)} new approved formulations.")

                if len(all_approved_this_run) >= config['max_approved_formulations']:
                    status.update(label="✅ Analysis complete: Target reached.", state="complete")
                    break
            else:
                status.update(label="✅ Analysis complete: All rounds finished.", state="complete")

        return self.combine_negotiation_results(all_approved_this_run, config, round_results.get('strategy', {}))

    def combine_negotiation_results(self, all_approved, config, final_strategy):
        """Consolidates results from all rounds into a final report."""
        unique_approved = self.ai_engine._remove_duplicate_formulations(all_approved)
        unique_approved.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return {
            'approved_formulations': unique_approved[:config['max_approved_formulations']],
            'strategy': final_strategy,
            'search_metrics': {
                'formulations_generated': len(unique_approved),
                'negotiation_rounds': config['max_negotiation_rounds']
            }
        }

    def display_results(self, results: Dict):
        """Renders the final results report in the main UI."""
        st.markdown("---")
        st.header("📊 Final Report")

        strategy = results.get('strategy', {})
        if strategy and isinstance(strategy, dict):
            with st.expander("📋 Final AI Strategy Overview", expanded=False):
                st.write(f"**Material Class:** `{strategy.get('material_class', 'N/A')}`")
                target_props = strategy.get('target_properties', {})
                if target_props and isinstance(target_props, dict):
                    st.write("**🎯 Final Target Properties:**")
                    prop_df = pd.DataFrame.from_dict(target_props, orient='index').fillna('-')
                    st.dataframe(prop_df, use_container_width=True)

        approved = results.get('approved_formulations', [])
        st.subheader(f"✅ Top {len(approved)} Approved Formulations")

        if not approved:
            st.warning("No formulations met the final approval criteria. Try relaxing the search parameters.")
        else:
            for i, formulation in enumerate(approved):
                self.display_formulation_card(i, formulation)
    
    def display_formulation_card(self, index: int, f: Dict):
        """IMPROVED: Renders a detailed, visually rich card for a single formulation."""
        status_badge = "approved-badge"
        status_text = "APPROVED"
        
        with st.container():
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([4, 1, 1.2])
            with c1:
                st.subheader(f'Formulation #{index + 1}')
                st.caption(f"**Agent:** {f.get('agent', 'N/A')} | **Found in Round:** {f.get('negotiation_round', 0) + 1} | **Risk Level:** {f.get('risk_level', 'N/A').title()}")
            with c2:
                st.metric("Score", f"{f.get('score', 0):.2f}")
            with c3:
                st.markdown(f'<div style="margin-top: 1.5rem; text-align: center;" class="{status_badge}">{status_text}</div>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["**Composition**", "**Predicted Properties**", "**AI Analysis**"])
            
            with tab1:
                self._render_composition_tab(f)
            with tab2:
                self._render_properties_tab(f)
            with tab3:
                self._render_analysis_tab(f)
            st.markdown('</div>', unsafe_allow_html=True)
            
    def _render_composition_tab(self, f: Dict):
        compounds = f.get('compounds', [])
        mass_ratios = f.get('ratios', [])
        mole_ratios = f.get('mole_ratios', [])
        
        comp_data = []
        for i, c in enumerate(compounds):
            comp_data.append({
                "Compound": getattr(c, 'iupac_name', f'Unknown (CID: {getattr(c, "cid", "N/A")})'),
                "Mass %": f"{mass_ratios[i]*100:.1f}%" if mass_ratios else "-",
                "Mole %": f"{mole_ratios[i]*100:.1f}%" if mole_ratios else "-",
                "MW (g/mol)": getattr(c, 'molecular_weight', '-'),
                "Formula": f"`{getattr(c, 'molecular_formula', '-')}`"
            })
        st.table(pd.DataFrame(comp_data))

    def _render_properties_tab(self, f: Dict):
        props = f.get('predicted_properties', {})
        if not props:
            st.info("No properties were predicted for this formulation.")
            return
        
        prop_list = []
        for prop, data in props.items():
            value = data.get('value', 'N/A')
            unit = data.get('unit', '')
            conf = data.get('confidence', 0)
            prop_list.append(f"**{prop.replace('_', ' ').title()}:** `{value:.3f} {unit}` (Confidence: {conf:.0%})")
        st.markdown("\n\n".join(f"- {p}" for p in prop_list))

    def _render_analysis_tab(self, f: Dict):
        ai_decision = f.get('ai_decision', {})
        reasons = ai_decision.get('reasons', [])
        st.info(f"**Summary:** {' | '.join(reasons)}")

        issues = f.get('feasibility', {}).get('compatibility_issues', [])
        if issues:
            st.error(f"**Potential Compatibility Issues:** {', '.join(issues)}")
        else:
            st.success("**Compatibility Check:** No major issues identified.")

def main():
    app = AdvancedMaterialsApp()
    config = app.render_sidebar()
    challenge_text = app.render_main_interface()
    
    if st.button("🚀 START ANALYSIS", use_container_width=True, disabled=not config.get('gemini_key')):
        if challenge_text:
            results = app.run_full_analysis(challenge_text, config)
            st.session_state.results = results
        else:
            st.warning("⚠️ Please enter a challenge description first.")

    if 'results' in st.session_state:
        app.display_results(st.session_state.results)

if __name__ == "__main__":
    main()

