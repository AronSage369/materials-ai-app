import streamlit as st
import json
import time
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ai_engine import MaterialsAIEngine
from pubchem_manager import PubChemManager
from property_predictor import AdvancedPropertyPredictor
from mixture_predictor import MixturePredictor
from compatibility_checker import CompatibilityChecker

class AdvancedMaterialsApp:
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
            st.session_state.ai_engine = None
            st.session_state.pubchem_manager = None
            st.session_state.property_predictor = None
            st.session_state.mixture_predictor = None
            st.session_state.compatibility_checker = None
            st.session_state.analysis_results = None
            st.session_state.current_round = 0
            st.session_state.total_formulations = 0

    def initialize_components(self, api_key: str):
        """Initialize all backend components"""
        try:
            st.session_state.ai_engine = MaterialsAIEngine(api_key)
            st.session_state.pubchem_manager = PubChemManager()
            st.session_state.property_predictor = AdvancedPropertyPredictor()
            st.session_state.mixture_predictor = MixturePredictor()
            st.session_state.compatibility_checker = CompatibilityChecker()
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            return False

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar and collect configuration"""
        st.sidebar.title("⚙️ Configuration")
        
        # API Configuration
        st.sidebar.subheader("🔑 API Configuration")
        api_key = st.sidebar.text_input("Google Gemini API Key", type="password", 
                                       help="Get your API key from Google AI Studio")
        
        # Analysis Parameters
        st.sidebar.subheader("🎯 Analysis Parameters")
        material_type = st.sidebar.selectbox(
            "Material Type",
            ["Solvent", "Coolant", "Absorbent", "Catalyst", "Polymer", "Lubricant", "Other"]
        )
        
        max_negotiation_rounds = st.sidebar.slider("Max Negotiation Rounds", 1, 5, 3)
        target_formulations = st.sidebar.slider("Target Formulations", 1, 20, 5)
        min_approval_score = st.sidebar.slider("Minimum Approval Score", 0.1, 1.0, 0.7)
        innovation_factor = st.sidebar.slider("Innovation Factor", 0.0, 1.0, 0.3)
        
        config = {
            'api_key': api_key,
            'material_type': material_type,
            'max_negotiation_rounds': max_negotiation_rounds,
            'target_formulations': target_formulations,
            'min_approval_score': min_approval_score,
            'innovation_factor': innovation_factor
        }
        
        return config

    def render_main_interface(self):
        """Render the main interface"""
        st.title("🧪 Advanced Materials Discovery AI")
        st.markdown("### Multi-Agent Adaptive Formulation System v2.0")
        
        # Challenge input
        st.subheader("🎯 Describe Your Material Challenge")
        challenge_text = st.text_area(
            "Be specific about your requirements:",
            height=120,
            placeholder="Example: 'I need a high-performance dielectric coolant for electronics with thermal conductivity > 0.5 W/m·K, flash point > 150°C, low toxicity, and good chemical stability. Budget is constrained.'"
        )
        
        return challenge_text

    def run_full_analysis(self, challenge_text: str, config: Dict[str, Any]):
        """Execute the full adaptive negotiation analysis"""
        if not challenge_text.strip():
            st.error("Please enter a material challenge description.")
            return None

        # Initialize components if needed
        if not st.session_state.initialized:
            if not self.initialize_components(config['api_key']):
                return None

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        round_container = st.container()

        all_approved_formulations = []
        
        with round_container:
            st.subheader("🔄 Adaptive Negotiation Progress")
            
            for round_num in range(config['max_negotiation_rounds']):
                st.session_state.current_round = round_num + 1
                current_progress = (round_num / config['max_negotiation_rounds']) * 100
                progress_bar.progress(int(current_progress))
                
                # Update status
                status_text.text(f"🔄 Round {round_num + 1}/{config['max_negotiation_rounds']}: Generating formulations...")
                
                # Step 1: Interpret challenge with relaxation
                relaxation_level = self._get_relaxation_level(round_num)
                strategy = st.session_state.ai_engine.interpret_challenge(
                    challenge_text, config['material_type'], relaxation_level
                )
                
                if not strategy:
                    st.error("Failed to interpret challenge. Please try again with more specific requirements.")
                    return None

                # Step 2: Find compounds
                status_text.text(f"🔍 Round {round_num + 1}: Searching for compounds...")
                compounds = st.session_state.pubchem_manager.find_compounds(
                    strategy, config['material_type']
                )
                
                if not compounds:
                    st.warning(f"No compounds found in round {round_num + 1}. Relaxing constraints...")
                    continue

                # Step 3: Generate formulations
                status_text.text(f"🤖 Round {round_num + 1}: Multi-agent formulation generation...")
                formulations = st.session_state.ai_engine.multi_agent_formulation_generation(
                    compounds, strategy, config['innovation_factor']
                )
                
                if not formulations:
                    st.warning(f"No formulations generated in round {round_num + 1}. Trying next round...")
                    continue

                # Step 4: Predict properties
                status_text.text(f"📊 Round {round_num + 1}: Predicting properties...")
                formulations = st.session_state.property_predictor.predict_all_properties(
                    formulations, strategy.get('target_properties', {})
                )

                # Step 5: Validate compatibility
                status_text.text(f"⚠️ Round {round_num + 1}: Safety analysis...")
                formulations = st.session_state.compatibility_checker.validate_all_formulations(formulations)

                # Step 6: Adaptive evaluation
                status_text.text(f"🎯 Round {round_num + 1}: Evaluating formulations...")
                approved_formulations = st.session_state.ai_engine.adaptive_evaluation(
                    formulations, strategy, round_num, config['min_approval_score']
                )
                
                # Display round results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Round {round_num + 1} Formulations", len(formulations))
                with col2:
                    st.metric("Approved", len(approved_formulations))
                with col3:
                    st.metric("Relaxation Level", relaxation_level)
                
                all_approved_formulations.extend(approved_formulations)
                st.session_state.total_formulations += len(formulations)
                
                # Check if we have enough formulations
                if len(all_approved_formulations) >= config['target_formulations']:
                    all_approved_formulations = all_approved_formulations[:config['target_formulations']]
                    status_text.text("✅ Target formulations reached!")
                    break
                
                if round_num < config['max_negotiation_rounds'] - 1:
                    st.info(f"🔄 Proceeding to round {round_num + 2} with relaxed constraints...")

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return {
            'strategy': strategy,
            'approved_formulations': all_approved_formulations,
            'total_rounds': st.session_state.current_round,
            'total_formulations_generated': st.session_state.total_formulations
        }

    def _get_relaxation_level(self, round_num: int) -> str:
        """Get relaxation level based on negotiation round"""
        levels = [
            "Standard constraints - focus on exact matches",
            "Mild relaxation - allow minor deviations from ideal targets", 
            "Moderate relaxation - consider broader property ranges",
            "Significant relaxation - prioritize feasibility over perfection",
            "Maximum relaxation - explore unconventional solutions"
        ]
        return levels[min(round_num, len(levels) - 1)]

    def display_results(self, results: Dict[str, Any]):
        """Display comprehensive results"""
        if not results or not results['approved_formulations']:
            st.warning("No suitable formulations found. Try relaxing your constraints or providing more specific requirements.")
            return

        st.success(f"🎉 Found {len(results['approved_formulations'])} approved formulations "
                  f"across {results['total_rounds']} rounds "
                  f"(from {results['total_formulations_generated']} total candidates)")

        # Display AI Strategy
        with st.expander("🧠 AI Analysis Strategy", expanded=True):
            strategy = results['strategy']
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Primary Objective:**", strategy.get('primary_objective', 'N/A'))
                st.write("**Key Constraints:**", strategy.get('key_constraints', 'N/A'))
                st.write("**Search Strategy:**", strategy.get('search_strategy', 'N/A'))
            
            with col2:
                st.write("**Success Metrics:**", strategy.get('success_metrics', 'N/A'))
                st.write("**Risk Tolerance:**", strategy.get('risk_tolerance', 'N/A'))

        # Target Properties Table
        if 'target_properties' in strategy:
            st.subheader("🎯 Target Properties")
            prop_data = []
            for prop, criteria in strategy['target_properties'].items():
                prop_data.append({
                    'Property': prop,
                    'Target': criteria.get('target', 'N/A'),
                    'Min': criteria.get('min', 'N/A'),
                    'Max': criteria.get('max', 'N/A'),
                    'Importance': criteria.get('importance', 'Medium')
                })
            st.table(pd.DataFrame(prop_data))

        # Formulation Comparison Visualization
        self._display_comparison_view(results['approved_formulations'])

        # Individual Formulation Cards
        st.subheader("🏆 Top Formulations")
        for i, formulation in enumerate(results['approved_formulations']):
            self.display_formulation_card(i + 1, formulation)

    def _display_comparison_view(self, formulations: List[Dict]):
        """Display parallel coordinates plot for formulation comparison"""
        if len(formulations) < 2:
            return

        # Prepare data for visualization
        plot_data = []
        for i, form in enumerate(formulations):
            row = {'Formulation': f'Form {i+1}', 'Approval_Score': form.get('approval_score', 0)}
            
            # Add key properties
            props = form.get('predicted_properties', {})
            for prop, value in props.items():
                if isinstance(value, (int, float)):
                    row[prop] = value
            
            plot_data.append(row)
        
        if len(plot_data) > 1:
            df = pd.DataFrame(plot_data)
            dimensions = [{'label': col, 'values': df[col]} for col in df.columns if col != 'Formulation']
            
            if dimensions:
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(color=df['Approval_Score'], 
                                 colorscale='Viridis',
                                 showscale=True,
                                 colorbar=dict(title='Approval Score')),
                        dimensions=dimensions
                    )
                )
                fig.update_layout(title="Formulation Comparison - Parallel Coordinates")
                st.plotly_chart(fig, use_container_width=True)

    def display_formulation_card(self, index: int, formulation: Dict):
        """Display detailed formulation card"""
        with st.container():
            st.markdown(f"### 🥇 Formulation #{index}")
            
            # Header with key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Approval Score", f"{formulation.get('approval_score', 0):.2f}")
            with col2:
                st.metric("Components", len(formulation.get('composition', [])))
            with col3:
                risk_score = formulation.get('compatibility_risk', 0)
                risk_color = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.7 else "🔴"
                st.metric("Risk Level", f"{risk_color} {risk_score:.2f}")
            with col4:
                st.metric("Agent", formulation.get('agent_type', 'Unknown'))
            
            # Tabs for detailed information
            tab1, tab2, tab3, tab4 = st.tabs(["🧪 Composition", "📊 Properties", "⚠️ Safety", "🤖 AI Analysis"])
            
            with tab1:
                self._display_composition(formulation)
            
            with tab2:
                self._display_properties(formulation)
            
            with tab3:
                self._display_safety_analysis(formulation)
            
            with tab4:
                self._display_ai_analysis(formulation)
            
            st.markdown("---")

    def _display_composition(self, formulation: Dict):
        """Display composition information"""
        composition = formulation.get('composition', [])
        if not composition:
            st.write("No composition data available.")
            return
        
        # Mass and mole percentages
        mass_data = []
        mole_data = []
        
        for comp in composition:
            mass_data.append({
                'Component': comp.get('name', 'Unknown'),
                'Mass %': f"{comp.get('mass_percentage', 0):.1f}%",
                'PubChem CID': comp.get('cid', 'N/A')
            })
            mole_data.append({
                'Component': comp.get('name', 'Unknown'),
                'Mole %': f"{comp.get('mole_percentage', 0):.1f}%",
                'Ratio': comp.get('mole_ratio', 'N/A')
            })
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mass Percentage**")
            st.table(pd.DataFrame(mass_data))
        with col2:
            st.write("**Mole Percentage**")
            st.table(pd.DataFrame(mole_data))

    def _display_properties(self, formulation: Dict):
        """Display predicted properties"""
        properties = formulation.get('predicted_properties', {})
        if not properties:
            st.write("No property predictions available.")
            return
        
        prop_data = []
        for prop, value in properties.items():
            prop_data.append({
                'Property': prop,
                'Value': f"{value:.4f}" if isinstance(value, float) else str(value),
                'Confidence': 'High' if formulation.get('confidence', 0) > 0.8 else 'Medium' if formulation.get('confidence', 0) > 0.5 else 'Low'
            })
        
        st.table(pd.DataFrame(prop_data))

    def _display_safety_analysis(self, formulation: Dict):
        """Display safety and compatibility analysis"""
        risk_score = formulation.get('compatibility_risk', 0)
        issues = formulation.get('compatibility_issues', [])
        
        # Risk indicator
        if risk_score < 0.3:
            st.success("🟢 Low Risk - Formulation appears chemically stable")
        elif risk_score < 0.7:
            st.warning("🟡 Medium Risk - Some compatibility concerns")
        else:
            st.error("🔴 High Risk - Significant compatibility issues")
        
        st.write(f"**Overall Risk Score:** {risk_score:.2f}")
        
        if issues:
            st.write("**Identified Issues:**")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.info("No major compatibility issues detected")

    def _display_ai_analysis(self, formulation: Dict):
        """Display AI reasoning and evaluation"""
        reasoning = formulation.get('reasoning', 'No reasoning provided.')
        strengths = formulation.get('strengths', [])
        limitations = formulation.get('limitations', [])
        
        st.write("**AI Reasoning:**")
        st.write(reasoning)
        
        col1, col2 = st.columns(2)
        with col1:
            if strengths:
                st.write("**Strengths:**")
                for strength in strengths:
                    st.write(f"✅ {strength}")
        
        with col2:
            if limitations:
                st.write("**Limitations:**")
                for limitation in limitations:
                    st.write(f"⚠️ {limitation}")

def main():
    app = AdvancedMaterialsApp()
    
    # Render sidebar and get configuration
    config = app.render_sidebar()
    
    # Render main interface and get challenge
    challenge_text = app.render_main_interface()
    
    # Analysis execution
    if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
        if not config['api_key']:
            st.error("Please enter your Gemini API key to proceed.")
            return
        
        with st.spinner("Initializing advanced materials discovery system..."):
            results = app.run_full_analysis(challenge_text, config)
            app.display_results(results)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Advanced Materials Discovery AI v2.0** | "
        "Multi-Agent Adaptive Formulation System | "
        "Built for Accelerated Materials Research"
    )

if __name__ == "__main__":
    main()
