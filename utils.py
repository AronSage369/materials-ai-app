# utils.py - Helper functions
import pubchempy as pcp
import pandas as pd

def get_compound_details(cid):
    """Get detailed compound information"""
    try:
        compound = pcp.Compound.from_cid(cid)
        return {
            'name': compound.iupac_name,
            'formula': compound.molecular_formula,
            'weight': compound.molecular_weight,
            'smiles': compound.canonical_smiles
        }
    except:
        return None

def estimate_properties(compound):
    """Simple property estimation"""
    properties = {}
    
    if compound.molecular_weight:
        # Simple estimations based on molecular weight
        properties['estimated_thermal_cond'] = 0.08 + (compound.molecular_weight / 10000)
        properties['estimated_viscosity'] = max(1, compound.molecular_weight / 100)
        
    return properties