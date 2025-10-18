import streamlit as st
import json
import time
import logging
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utils first with minimal dependencies and better error handling
try:
    from utils import CacheManager, MemoryManager, data_validator
    UTILS_AVAILABLE = True
    logger.info("Utils imported successfully")
except ImportError as e:
    logger.error(f"Failed to import utils: {e}")
    UTILS_AVAILABLE = False
    # Create minimal fallbacks
    class CacheManager:
        def get(self, key): return None
        def set(self, key, value): pass
        def clear(self): pass
    class MemoryManager:
        @staticmethod
        def cleanup_memory(): pass
        @staticmethod
        def get_memory_usage(): return 0.0
    class data_validator:
        @staticmethod
        def validate_formulation(formulation): return True, "Fallback validation"
        @staticmethod
        def validate_component(component): return True, "Fallback validation"

# Try to import all components with fallbacks
COMPONENT_AVAILABILITY = {}

try:
    from ai_strategist import AIStrategist
    COMPONENT_AVAILABILITY['ai_strategist'] = True
except ImportError as e:
    logger.error(f"Failed to import AIStrategist: {e}")
    COMPONENT_AVAILABILITY['ai_strategist'] = False

try:
    from ai_engine import CreativeAIEngine
    COMPONENT_AVAILABILITY['creative_engine'] = True
except ImportError as e:
    logger.error(f"Failed to import CreativeAIEngine: {e}")
    COMPONENT_AVAILABILITY['creative_engine'] = False

try:
    from computational_predictor import ComputationalPredictor
    COMPONENT_AVAILABILITY['computational_predictor'] = True
except ImportError as e:
    logger.error(f"Failed to import ComputationalPredictor: {e}")
    COMPONENT_AVAILABILITY['computational_predictor'] = False

try:
    from pubchem_manager import PubChemManager
    COMPONENT_AVAILABILITY['pubchem_manager'] = True
except ImportError as e:
    logger.error(f"Failed to import PubChemManager: {e}")
    COMPONENT_AVAILABILITY['pubchem_manager'] = False

try:
    from property_predictor import AdvancedPropertyPredictor
    COMPONENT_AVAILABILITY['property_predictor'] = True
except ImportError as e:
    logger.error(f"Failed to import AdvancedPropertyPredictor: {e}")
    COMPONENT_AVAILABILITY['property_predictor'] = False

try:
    from compatibility_checker import CompatibilityChecker
    COMPONENT_AVAILABILITY['compatibility_checker'] = True
except ImportError as e:
    logger.error(f"Failed to import CompatibilityChecker: {e}")
    COMPONENT_AVAILABILITY['compatibility_checker'] = False

class AdvancedMaterialsDiscoveryApp:
    def __init__(self):
        st.set_page_config(
            page_title="Advanced Materials Discovery AI",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.ai_strategist = None
            st.session_state.creative_engine = None
            st.session_state.computational_predictor = None
            st.session_state.pubchem_manager = None
            st.session_state.property_predictor = None
            st.session_state.compatibility_checker = None
            st.session_state.analysis_results = None
            st.session_state.current_step = 0
            st.session_state.api_key = ""

    def initialize_components(self, api_key: str) -> bool:
        """Initialize AI components with the provided API key"""
        try:
            if not api_key:
                st.error("🔑 Please enter your Gemini API key")
                return False
            
            # Show initialization progress
            with st.spinner("🔄 Initializing AI components..."):
                # Initialize AI Strategist
                if COMPONENT_AVAILABILITY.get('ai_strategist', False):
                    try:
                        st.session_state.ai_strategist = AIStrategist(api_key)
                        st.success("✅ AI Strategist initialized")
                    except Exception as e:
                        st.error(f"❌ AI Strategist initialization failed: {e}")
                        return False
                else:
                    st.error("❌ AI Strategist not available")
                    return False
                
                # Initialize Creative AI Engine
                if COMPONENT_AVAILABILITY.get('creative_engine', False):
                    try:
                        st.session_state.creative_engine = CreativeAIEngine(api_key)
                        st.success("✅ Creative AI Engine initialized")
                    except Exception as e:
                        st.error(f"❌ Creative AI Engine initialization failed: {e}")
                        return False
                else:
                    st.error("❌ Creative AI Engine not available")
                    return False
                
                # Initialize other components (optional)
                if COMPONENT_AVAILABILITY.get('computational_predictor', False):
                    try:
                        st.session_state.computational_predictor = ComputationalPredictor()
                        st.success("✅ Computational Predictor initialized")
                    except Exception as e:
                        st.warning(f"⚠️ Computational Predictor initialization failed: {e}")
                
                if COMPONENT_AVAILABILITY.get('pubchem_manager', False):
                    try:
                        st.session_state.pubchem_manager = PubChemManager()
                        st.success("✅ PubChem Manager initialized")
                    except Exception as e:
                        st.warning(f"⚠️ PubChem Manager initialization failed: {e}")
                
                if COMPONENT_AVAILABILITY.get('property_predictor', False):
                    try:
                        st.session_state.property_predictor = AdvancedPropertyPredictor()
                        st.success("✅ Property Predictor initialized")
                    except Exception as e:
                        st.warning(f"⚠️ Property Predictor initialization failed: {e}")
                
                if COMPONENT_AVAILABILITY.get('compatibility_checker', False):
                    try:
                        st.session_state.compatibility_checker = CompatibilityChecker()
                        st.success("✅ Compatibility Checker initialized")
                    except Exception as e:
                        st.warning(f"⚠️ Compatibility Checker initialization failed: {e}")
            
            st.session_state.initialized = True
            st.session_state.api_key = api_key
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to initialize AI components: {str(e)}")
            return False

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("🧪 Materials Discovery AI")
        st.sidebar.markdown("Configure your materials discovery challenge")
        
        # Show dependency status
        st.sidebar.markdown("### 🔧 Component Status")
        for component, available in COMPONENT_AVAILABILITY.items():
            status = "✅" if available else "❌"
            st.sidebar.text(f"{status} {component.replace('_', ' ').title()}")
        
        # API Key input
        api_key = st.sidebar.text_input(
            "🔑 Gemini API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            help="Enter your Google Gemini API key"
        )
        
        if api_key and api_key != st.session_state.get('api_key', ''):
            if self.initialize_components(api_key):
                st.sidebar.success("✅ Components initialized!")
            else:
                st.sidebar.error("❌ Initialization failed")
        
        # Material type selection
        material_types = ['solvent', 'coolant', 'absorbent', 'catalyst', 'polymer', 'lubricant']
        material_type = st.sidebar.selectbox(
            "📦 Material Type",
            material_types,
            help="Select the type of material you want to discover"
        )
        
        # Innovation factor
        innovation_factor = st.sidebar.slider(
            "💡 Innovation Factor",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values generate more innovative formulations"
        )
        
        # Minimum approval score
        min_approval_score = st.sidebar.slider(
            "🎯 Minimum Approval Score",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Minimum score for formulations to be considered"
        )
        
        return {
            'material_type': material_type,
            'innovation_factor': innovation_factor,
            'min_approval_score': min_approval_score
        }

    def render_main_interface(self):
        """Render the main interface"""
        st.title("🚀 Advanced Materials Discovery AI")
        st.markdown("""
        Discover innovative material formulations using AI-powered computational chemistry and creative thinking.
        
        **How it works:**
        1. 🧠 AI analyzes your challenge and develops a scientific strategy
        2. 🔍 Searches PubChem for relevant compounds
        3. 🤖 Creative AI agents design innovative formulations
        4. 📊 Computational models predict properties
        5. ⚠️ Compatibility checker ensures safety
        6. 🎯 AI evaluates and ranks the best formulations
        """)
        
        # Show component availability status
        st.subheader("🔧 System Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI Strategist", "✅" if COMPONENT_AVAILABILITY.get('ai_strategist', False) else "❌")
            st.metric("Creative Engine", "✅" if COMPONENT_AVAILABILITY.get('creative_engine', False) else "❌")
        with col2:
            st.metric("PubChem Search", "✅" if COMPONENT_AVAILABILITY.get('pubchem_manager', False) else "❌")
            st.metric("Property Predictor", "✅" if COMPONENT_AVAILABILITY.get('property_predictor', False) else "❌")
        with col3:
            st.metric("Computational Predictor", "✅" if COMPONENT_AVAILABILITY.get('computational_predictor', False) else "❌")
            st.metric("Compatibility Checker", "✅" if COMPONENT_AVAILABILITY.get('compatibility_checker', False) else "❌")
        
        # Check if core components are available
        if not COMPONENT_AVAILABILITY.get('ai_strategist', False) or not COMPONENT_AVAILABILITY.get('creative_engine', False):
            st.error("""
            ❌ Core AI components are not available. Please check:
            - Ensure all Python files are in the same directory
            - Check that dependencies are installed
            - Verify there are no syntax errors in the module files
            """)
            return None, None
        
        # Challenge input
        challenge_text = st.text_area(
            "🎯 Describe Your Materials Challenge",
            height=100,
            placeholder="e.g., 'I need a biodegradable solvent with high thermal stability for electronics cleaning that is non-toxic and has low vapor pressure...'",
            help="Be specific about your requirements, constraints, and desired properties"
        )
        
        # Target properties
        st.subheader("🎯 Target Properties")
        col1, col2, col3 = st.columns(3)
        
        target_properties = {}
        with col1:
            if st.checkbox("Thermal Conductivity", value=True):
                target_properties['thermal_conductivity'] = {
                    'min': st.number_input("Min (W/m·K)", value=0.1, key='tc_min'),
                    'max': st.number_input("Max (W/m·K)", value=1.0, key='tc_max')
                }
            if st.checkbox("Viscosity", value=True):
                target_properties['viscosity'] = {
                    'min': st.number_input("Min (cP)", value=1.0, key='vis_min'),
                    'max': st.number_input("Max (cP)", value=100.0, key='vis_max')
                }
        
        with col2:
            if st.checkbox("Boiling Point", value=True):
                target_properties['boiling_point'] = {
                    'min': st.number_input("Min (°C)", value=50.0, key='bp_min'),
                    'max': st.number_input("Max (°C)", value=300.0, key='bp_max')
                }
            if st.checkbox("Density", value=True):
                target_properties['density'] = {
                    'min': st.number_input("Min (g/mL)", value=0.8, key='den_min'),
                    'max': st.number_input("Max (g/mL)", value=1.5, key='den_max')
                }
        
        with col3:
            if st.checkbox("Surface Tension"):
                target_properties['surface_tension'] = {
                    'min': st.number_input("Min (mN/m)", value=20.0, key='st_min'),
                    'max': st.number_input("Max (mN/m)", value=80.0, key='st_max')
                }
            if st.checkbox("Specific Heat"):
                target_properties['specific_heat'] = {
                    'min': st.number_input("Min (J/g·K)", value=1.0, key='sh_min'),
                    'max': st.number_input("Max (J/g·K)", value=4.0, key='sh_max')
                }
        
        return challenge_text, target_properties

    def run_advanced_analysis(self, challenge_text: str, config: Dict, target_properties: Dict) -> Dict[str, Any]:
        """Run advanced AI-driven analysis"""
        if not st.session_state.initialized:
            st.error("❌ Please initialize AI components first")
            return None
        
        if not challenge_text.strip():
            st.error("❌ Please provide a materials challenge description")
            return None
            
        results = {}
        
        # Step 1: AI Strategic Thinking
        with st.spinner("🧠 AI is thinking deeply about your challenge..."):
            try:
                strategy = st.session_state.ai_strategist.think_about_challenge(
                    challenge_text, config['material_type']
                )
                results['strategy'] = strategy
                st.success("✅ AI strategy developed")
            except Exception as e:
                st.error(f"❌ AI thinking failed: {e}")
                return None
        
        # Step 2: Intelligent Compound Search
        with st.spinner("🔍 Searching for relevant compounds..."):
            try:
                if (COMPONENT_AVAILABILITY.get('pubchem_manager', False) and 
                    st.session_state.pubchem_manager is not None):
                    search_terms = (strategy.get('search_strategy', {})
                                  .get('primary_terms', []) + 
                                  strategy.get('search_strategy', {})
                                  .get('innovative_terms', []))
                    
                    compounds = st.session_state.pubchem_manager.find_compounds(
                        strategy, config['material_type']
                    )
                else:
                    st.warning("⚠️ PubChem Manager not available, using fallback compounds")
                    compounds = self._get_fallback_compounds(config['material_type'])
                
                results['compounds'] = compounds
                st.success(f"✅ Found {len(compounds)} compounds")
            except Exception as e:
                st.error(f"❌ Compound search failed: {e}")
                return None
        
        # Step 3: Creative Formulation Generation
        with st.spinner("🤖 Creative AI agents are designing formulations..."):
            try:
                formulations = st.session_state.creative_engine.creative_formulation_generation(
                    compounds, strategy, config['innovation_factor']
                )
                results['formulations'] = formulations
                st.success(f"✅ Generated {len(formulations)} formulations")
            except Exception as e:
                st.error(f"❌ Formulation generation failed: {e}")
                return None
        
        # Step 4: Advanced Property Prediction
        with st.spinner("📊 Running computational predictions..."):
            try:
                enhanced_target_props = self._extract_target_properties(strategy, target_properties)
                
                # Property prediction (optional)
                if (COMPONENT_AVAILABILITY.get('property_predictor', False) and 
                    st.session_state.property_predictor is not None):
                    formulations = st.session_state.property_predictor.predict_all_properties(
                        formulations, enhanced_target_props
                    )
                
                # Computational predictions (optional)
                if (COMPONENT_AVAILABILITY.get('computational_predictor', False) and 
                    st.session_state.computational_predictor is not None):
                    for formulation in formulations:
                        try:
                            comp_predictions = st.session_state.computational_predictor.predict_advanced_properties(
                                formulation, enhanced_target_props
                            )
                            formulation['computational_predictions'] = comp_predictions
                        except Exception as e:
                            st.warning(f"Computational prediction failed for one formulation: {e}")
                            formulation['computational_predictions'] = {}
                
                results['formulations'] = formulations
                st.success("✅ Property predictions completed")
            except Exception as e:
                st.error(f"❌ Property prediction failed: {e}")
                return None
        
        # Step 5: Compatibility Analysis
        with st.spinner("⚠️ Analyzing chemical compatibility..."):
            try:
                if (COMPONENT_AVAILABILITY.get('compatibility_checker', False) and 
                    st.session_state.compatibility_checker is not None):
                    formulations = st.session_state.compatibility_checker.validate_all_formulations(formulations)
                else:
                    # Add basic compatibility info
                    for formulation in formulations:
                        formulation['compatibility_risk'] = 0.3
                        formulation['compatibility_warnings'] = ["Compatibility check skipped"]
                        formulation['compatibility_summary'] = "Unknown"
                
                results['formulations'] = formulations
                st.success("✅ Compatibility analysis completed")
            except Exception as e:
                st.error(f"❌ Compatibility analysis failed: {e}")
                return None
        
        # Step 6: AI Evaluation and Ranking
        with st.spinner("🎯 AI is evaluating and ranking formulations..."):
            try:
                approved_formulations = self._ai_evaluation(formulations, strategy, config)
                results['approved_formulations'] = approved_formulations
                results['total_compounds_found'] = len(compounds)
                results['total_formulations_generated'] = len(formulations)
                results['ai_thinking'] = strategy.get('scientific_analysis', '')
                results['innovative_ideas'] = strategy.get('innovative_ideas', [])
                
                st.success(f"✅ Analysis complete! Found {len(approved_formulations)} promising formulations")
            except Exception as e:
                st.error(f"❌ AI evaluation failed: {e}")
                return None
        
        return results

    def _get_fallback_compounds(self, material_type: str) -> List[Dict]:
        """Provide fallback compounds when PubChem is not available"""
        fallback_compounds = {
            'solvent': [
                {'cid': 887, 'name': 'Methanol', 'molecular_weight': 32.04, 'smiles': 'CO', 'category': 'balanced'},
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                {'cid': 6344, 'name': 'Ethanol', 'molecular_weight': 46.07, 'smiles': 'CCO', 'category': 'balanced'},
                {'cid': 6579, 'name': 'Acetone', 'molecular_weight': 58.08, 'smiles': 'CC(=O)C', 'category': 'balanced'},
            ],
            'coolant': [
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                {'cid': 174, 'name': 'Ethylene Glycol', 'molecular_weight': 62.07, 'smiles': 'OCCO', 'category': 'specialist'},
                {'cid': 1030, 'name': 'Propylene Glycol', 'molecular_weight': 76.09, 'smiles': 'CC(O)CO', 'category': 'specialist'},
            ],
            'polymer': [
                {'cid': 6325, 'name': 'Polyethylene', 'molecular_weight': 28000, 'smiles': 'C=C', 'category': 'balanced'},
                {'cid': 76958, 'name': 'Polypropylene', 'molecular_weight': 42000, 'smiles': 'CC=C', 'category': 'balanced'},
            ],
            'absorbent': [
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                {'cid': 24823, 'name': 'Silica Gel', 'molecular_weight': 60.08, 'smiles': 'O=[Si]=O', 'category': 'specialist'},
            ],
            'catalyst': [
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                {'cid': 222, 'name': 'Ammonia', 'molecular_weight': 17.03, 'smiles': 'N', 'category': 'specialist'},
            ],
            'lubricant': [
                {'cid': 962, 'name': 'Water', 'molecular_weight': 18.02, 'smiles': 'O', 'category': 'balanced'},
                {'cid': 24748, 'name': 'Mineral Oil', 'molecular_weight': 400.0, 'smiles': 'CCCCCCCCCCCCCCCC', 'category': 'balanced'},
            ]
        }
        return fallback_compounds.get(material_type, fallback_compounds['solvent'])

    def _extract_target_properties(self, strategy: Dict, user_target_props: Dict) -> Dict[str, Any]:
        """Extract target properties from AI strategy and user input"""
        target_props = user_target_props.copy()
        
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
            try:
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
            except Exception as e:
                logger.warning(f"Evaluation failed for one formulation: {e}")
                continue
        
        # Sort by overall score
        approved.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        return approved

    def _calculate_computational_score(self, formulation: Dict) -> float:
        """Calculate score based on computational predictions"""
        comp_predictions = formulation.get('computational_predictions', {})
        if not comp_predictions:
            return 0.5
        
        # Average confidence of computational predictions
        try:
            confidences = [pred.get('confidence', 0) for pred in comp_predictions.values()]
            return sum(confidences) / len(confidences) if confidences else 0.5
        except:
            return 0.5

    def display_results(self, results: Dict[str, Any]):
        """Display analysis results"""
        if not results:
            return
        
        st.header("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Compounds Found", results['total_compounds_found'])
        with col2:
            st.metric("Formulations Generated", results['total_formulations_generated'])
        with col3:
            st.metric("Promising Formulations", len(results['approved_formulations']))
        with col4:
            if results['approved_formulations']:
                avg_score = sum([f.get('overall_score', 0) for f in results['approved_formulations']]) / len(results['approved_formulations'])
                st.metric("Average Score", f"{avg_score:.2f}")
            else:
                st.metric("Average Score", "N/A")
        
        # AI Strategy
        with st.expander("🧠 AI Strategy Analysis"):
            strategy = results['strategy']
            st.subheader("Scientific Analysis")
            st.write(strategy.get('scientific_analysis', 'No analysis available'))
            
            st.subheader("Key Mechanisms")
            for mechanism in strategy.get('key_mechanisms', []):
                st.write(f"• {mechanism}")
            
            st.subheader("Search Strategy")
            st.json(strategy.get('search_strategy', {}))
            
            st.subheader("Innovative Ideas")
            for idea in strategy.get('innovative_ideas', []):
                st.write(f"💡 {idea}")
        
        # Formulations
        st.subheader("🎯 Top Formulations")
        formulations = results['approved_formulations'][:10]  # Show top 10
        
        if not formulations:
            st.info("No formulations met the minimum approval score. Try adjusting your criteria or increasing the innovation factor.")
            return
            
        for i, formulation in enumerate(formulations, 1):
            with st.expander(f"Formulation #{i} | Score: {formulation.get('overall_score', 0):.2f} | {formulation.get('agent_type', 'Unknown')}"):
                self._display_formulation_details(formulation)
        
        # Visualization
        if len(formulations) > 1:
            self._create_visualizations(formulations)

    def _display_formulation_details(self, formulation: Dict):
        """Display detailed information about a formulation"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Composition")
            composition_data = []
            for comp in formulation['composition']:
                composition_data.append({
                    'Name': comp.get('name', 'Unknown'),
                    'Mass %': f"{comp.get('mass_percentage', 0):.1f}%",
                    'Molecular Weight': comp.get('molecular_weight', 'N/A')
                })
            composition_df = pd.DataFrame(composition_data)
            st.dataframe(composition_df, use_container_width=True)
            
            st.subheader("Properties")
            if 'predicted_properties' in formulation:
                props_data = [{'Property': k, 'Value': f"{v:.2f}" if isinstance(v, (int, float)) else str(v)} 
                            for k, v in formulation['predicted_properties'].items()]
                props_df = pd.DataFrame(props_data)
                st.dataframe(props_df, use_container_width=True)
        
        with col2:
            st.subheader("Scores & Evaluation")
            scores = {
                'Overall Score': formulation.get('overall_score', 0),
                'Innovation Score': formulation.get('innovation_score', 0),
                'Strategy Alignment': formulation.get('strategy_alignment', 0),
                'Compatibility Risk': formulation.get('compatibility_risk', 0)
            }
            
            for metric, score in scores.items():
                st.progress(score, text=f"{metric}: {score:.2f}")
            
            # Computational insights
            if formulation.get('computational_insights'):
                st.subheader("Computational Insights")
                st.json(formulation['computational_insights'])
            
            # Scientific evaluation
            if formulation.get('scientific_evaluation'):
                st.subheader("Scientific Evaluation")
                st.json(formulation['scientific_evaluation'])
            
            # Compatibility warnings
            if formulation.get('compatibility_warnings'):
                st.subheader("Compatibility Warnings")
                for warning in formulation['compatibility_warnings']:
                    if 'HIGH' in warning or 'EXTREME' in warning:
                        st.error(warning)
                    elif 'Medium' in warning:
                        st.warning(warning)
                    else:
                        st.info(warning)

    def _create_visualizations(self, formulations: List[Dict]):
        """Create visualizations for formulations"""
        st.subheader("📈 Formulation Analysis")
        
        # Score distribution
        scores = [f.get('overall_score', 0) for f in formulations]
        fig = px.histogram(x=scores, nbins=20, title="Score Distribution", 
                          labels={'x': 'Overall Score', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application runner"""
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Check if components are initialized
        if not st.session_state.initialized:
            st.info("👆 Please enter your Gemini API key in the sidebar to get started")
            return
        
        # Render main interface
        result = self.render_main_interface()
        if result is None:
            return
            
        challenge_text, target_properties = result
        
        # Run analysis button
        if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
            if not challenge_text.strip():
                st.error("❌ Please describe your materials challenge")
                return
            
            # Run the analysis
            results = self.run_advanced_analysis(challenge_text, config, target_properties)
            
            if results:
                # Store results in session state
                st.session_state.analysis_results = results
                
                # Display results
                self.display_results(results)
                
                # Download option
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name="materials_analysis_results.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.error("❌ Analysis failed. Please check your inputs and try again.")
        
        # Display previous results if available
        if st.session_state.analysis_results:
            st.header("📋 Previous Results")
            if st.button("Show Previous Results", use_container_width=True):
                self.display_results(st.session_state.analysis_results)

# Run the application
if __name__ == "__main__":
    app = AdvancedMaterialsDiscoveryApp()
    app.run()
