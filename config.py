import os
from typing import Dict, Any

class Config:
    """Configuration management for Materials Discovery AI"""
    
    # API Settings
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
    MAX_FORMULATIONS = int(os.getenv('MAX_FORMULATIONS', '20'))
    MAX_COMPOUNDS = int(os.getenv('MAX_COMPOUNDS', '100'))
    
    # Performance Settings
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '3600'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Safety Settings
    MAX_COMPATIBILITY_RISK = float(os.getenv('MAX_COMPATIBILITY_RISK', '0.8'))
    MIN_APPROVAL_SCORE = float(os.getenv('MIN_APPROVAL_SCORE', '0.6'))
    
    # App Settings
    PORT = int(os.getenv('PORT', '8501'))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    @classmethod
    def get_material_settings(cls, material_type: str) -> Dict[str, Any]:
        """Get settings specific to material type"""
        settings = {
            'solvent': {
                'max_components': 4, 
                'innovation_weight': 0.7,
                'search_terms': ['solvent', 'organic solvent', 'polar solvent']
            },
            'polymer': {
                'max_components': 3, 
                'innovation_weight': 0.6,
                'search_terms': ['polymer', 'plastic', 'resin']
            },
            'catalyst': {
                'max_components': 5, 
                'innovation_weight': 0.8,
                'search_terms': ['catalyst', 'enzyme', 'photocatalyst']
            },
            'coolant': {
                'max_components': 4,
                'innovation_weight': 0.5,
                'search_terms': ['coolant', 'heat transfer', 'thermal fluid']
            },
            'absorbent': {
                'max_components': 3,
                'innovation_weight': 0.7,
                'search_terms': ['absorbent', 'adsorbent', 'porous material']
            },
            'lubricant': {
                'max_components': 4,
                'innovation_weight': 0.6,
                'search_terms': ['lubricant', 'lubricating oil', 'grease']
            }
        }
        return settings.get(material_type.lower(), {
            'max_components': 4, 
            'innovation_weight': 0.7,
            'search_terms': [material_type, 'chemical compound']
        })
