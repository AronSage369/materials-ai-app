# app.py - SIMPLIFIED VERSION
import streamlit as st
import pubchempy as pcp
import pandas as pd
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Materials Discovery AI",
    page_icon="🧪",
    layout="wide"
)

def main():
    st.title("🧪 Materials Discovery AI")
    st.write("Discover optimal chemical formulations for any challenge!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_compounds = st.slider("Max compounds to analyze", 10, 50, 20)
        challenge_type = st.selectbox("Challenge Type", ["Coolant", "Adsorbent", "Catalyst", "Custom"])
    
    # Main input
    st.subheader("📝 Describe Your Challenge")
    challenge_text = st.text_area(
        "Paste your materials challenge:",
        height=150,
        placeholder="Example: Need coolant with high thermal conductivity, low viscosity, and flash point > 150°C"
    )
    
    if st.button("🚀 Analyze Materials", type="primary"):
        if challenge_text:
            with st.spinner("Searching PubChem and analyzing compounds..."):
                try:
                    # Simple analysis
                    results = analyze_materials(challenge_text, max_compounds)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    for i, compound in enumerate(results[:5]):
                        with st.expander(f"Result {i+1}: {compound.get('name', 'Unknown')}"):
                            st.write(f"**PubChem CID:** {compound.get('cid', 'N/A')}")
                            st.write(f"**Formula:** {compound.get('formula', 'N/A')}")
                            st.write(f"**Molecular Weight:** {compound.get('weight', 'N/A')}")
                            st.write(f"**IUPAC Name:** {compound.get('iupac_name', 'N/A')}")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a challenge description")

def analyze_materials(challenge_text, max_compounds):
    """Simple materials analysis function"""
    results = []
    
    # Define search terms based on challenge type
    search_terms = []
    challenge_lower = challenge_text.lower()
    
    if any(word in challenge_lower for word in ['coolant', 'cooling', 'thermal']):
        search_terms = ['alkanes', 'siloxane', 'ester', 'glycol', 'paraffin']
    elif any(word in challenge_lower for word in ['adsorbent', 'adsorption']):
        search_terms = ['zeolite', 'activated carbon', 'silica', 'MOF']
    else:
        search_terms = ['organic compound', 'polymer', 'industrial chemical']
    
    # Search PubChem
    for term in search_terms[:3]:  # Limit to 3 terms
        try:
            compounds = pcp.get_compounds(term, 'name', listkey_count=5)
            for compound in compounds:
                if len(results) < max_compounds:
                    results.append({
                        'cid': compound.cid,
                        'name': compound.iupac_name,
                        'formula': compound.molecular_formula,
                        'weight': compound.molecular_weight,
                        'iupac_name': compound.iupac_name
                    })
        except Exception as e:
            st.write(f"Note: Couldn't search '{term}': {e}")
    
    return results

if __name__ == "__main__":
    main()