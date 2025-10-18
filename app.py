# In your app.py, replace the run_analysis method with this:

def run_analysis(self, challenge_text, config):
    """Run adaptive multi-round analysis with negotiation"""
    if not config['gemini_key']:
        st.error("❌ Please enter your Gemini API key in the sidebar")
        return None
    
    try:
        self.ai_engine.set_api_key(config['gemini_key'])
        
        all_results = []
        max_rounds = 3  # Maximum negotiation rounds
        
        for round_num in range(max_rounds):
            st.write(f"---")
            st.subheader(f"🔄 Negotiation Round {round_num + 1}")
            
            # Phase 1: Adaptive challenge interpretation
            with st.spinner(f"Round {round_num + 1}: Interpreting challenge..."):
                strategy = self.ai_engine.interpret_challenge(challenge_text, config['material_type'], round_num)
            
            # Phase 2: Compound discovery
            with st.spinner(f"Round {round_num + 1}: Searching compounds..."):
                compounds_data = self.pubchem_manager.find_compounds(
                    strategy, 
                    config['max_compounds'],
                    config['search_depth']
                )
            
            # Phase 3: Multi-agent formulation generation
            with st.spinner(f"Round {round_num + 1}: Multi-agent formulation..."):
                formulations = self.ai_engine.multi_agent_formulation_generation(compounds_data, strategy)
            
            # Phase 4: Mixture prediction
            if config['enable_mixture_prediction'] and formulations:
                with st.spinner(f"Round {round_num + 1}: Predicting properties..."):
                    formulations = self.mixture_predictor.predict_all_properties(formulations, strategy)
            
            # Phase 5: Compatibility checking
            if config['enable_compatibility'] and formulations:
                with st.spinner(f"Round {round_num + 1}: Checking compatibility..."):
                    formulations = self.compatibility_checker.validate_all_formulations(formulations)
            
            # Phase 6: Adaptive evaluation
            with st.spinner(f"Round {round_num + 1}: Evaluating formulations..."):
                round_results = self.ai_engine.adaptive_evaluation(
                    formulations, strategy, config['min_confidence'], round_num
                )
            
            all_results.append(round_results)
            
            # Check if we have enough approved formulations
            approved_count = len(round_results.get('approved_formulations', []))
            if approved_count >= 5 or round_num == max_rounds - 1:
                st.success(f"✅ Completed {round_num + 1} negotiation rounds")
                break
            else:
                st.warning(f"⚠️ Only {approved_count} approved formulations. Starting next negotiation round...")
        
        # Combine results from all rounds
        final_results = self.combine_negotiation_results(all_results)
        return final_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

def combine_negotiation_results(self, all_results: List[Dict]) -> Dict[str, Any]:
    """Combine results from all negotiation rounds"""
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
    
    # Remove duplicates and sort
    all_approved = self.remove_duplicate_formulations(all_approved)
    all_rejected = self.remove_duplicate_formulations(all_rejected)
    
    all_approved.sort(key=lambda x: x.get('score', 0), reverse=True)
    all_rejected.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return {
        'approved_formulations': all_approved[:10],
        'rejected_formulations': all_rejected[:5],
        'search_metrics': total_metrics,
        'strategy': all_results[-1].get('strategy', {}) if all_results else {},
        'negotiation_summary': f"Completed {len(all_results)} rounds with {len(all_approved)} total approved"
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

# Also update the display_results method to show both approved and rejected:
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
    
    # Approved formulations
    approved = results.get('approved_formulations', [])
    if approved:
        st.subheader("✅ Approved Formulations")
        for i, formulation in enumerate(approved):
            self.display_formulation_card(i, formulation, config, approved=True)
    else:
        st.warning("❌ No formulations met the approval criteria")
    
    # Show top rejected formulations with explanations
    rejected = results.get('rejected_formulations', [])
    if rejected:
        st.subheader("⚠️ Promising Rejected Formulations")
        st.info("These formulations showed promise but didn't meet all criteria. Consider them for further investigation.")
        
        for i, formulation in enumerate(rejected):
            self.display_formulation_card(i, formulation, config, approved=False)
    
    # Strategy overview
    with st.expander("📋 AI Strategy Overview", expanded=True):
        strategy = results.get('strategy', {})
        st.write(f"**Material Class:** {strategy.get('material_class', 'Custom')}")
        st.write("**Final Target Properties:**")
        for prop, criteria in strategy.get('target_properties', {}).items():
            st.write(f"- {prop}: {criteria}")

# Update the display_formulation_card to show approval status:
def display_formulation_card(self, index, formulation, config, approved=True):
    """Display formulation card with approval status"""
    
    status_color = "✅" if approved else "⚠️"
    status_text = "APPROVED" if approved else "REJECTED"
    status_class = "approved-badge" if approved else "rejected-badge"
    
    with st.container():
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        
        # Header with status
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{status_color} Formulation #{index + 1} ({status_text})")
        
        with col2:
            score = formulation.get('score', 0)
            st.metric("Overall Score", f"{score:.2f}")
        
        with col3:
            st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        # Show which agent created this formulation
        agent = formulation.get('agent', 'unknown')
        st.write(f"**Created by:** {agent.capitalize()} Agent")
        
        # Rest of your existing display code...
        # [Keep your existing compound display code here]
        
        st.markdown('</div>', unsafe_allow_html=True)
