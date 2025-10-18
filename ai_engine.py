import random
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from utils import cached, MemoryManager

# Try to import dependencies with fallbacks
try:
    from ai_strategist import AIStrategist
except ImportError:
    logging.warning("AIStrategist not available, using fallback")
    from utils import AIStrategistFallback as AIStrategist

try:
    from computational_predictor import ComputationalPredictor
except ImportError:
    logging.warning("ComputationalPredictor not available, using fallback")
    from utils import ComputationalPredictorFallback as ComputationalPredictor

class CreativeAIEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        try:
            self.strategist = AIStrategist(api_key)
            self.computational_predictor = ComputationalPredictor()
        except Exception as e:
            self.logger.error(f"Failed to initialize AI components: {e}")
            raise
        
        self.creative_agents = self._initialize_creative_agents()
        
    def _initialize_creative_agents(self) -> Dict[str, Any]:
        """Initialize creative AI agents with different thinking styles"""
        return {
            'quantum_thinker': {
                'name': 'Quantum Thinking Agent',
                'focus': 'Electronic structure and quantum effects',
                'approach': 'Focuses on molecular orbitals, band structure, quantum confinement'
            },
            'biomimetic_thinker': {
                'name': 'Biomimetic Agent', 
                'focus': 'Nature-inspired solutions',
                'approach': 'Looks at biological systems and natural materials'
            },
            'combinatorial_thinker': {
                'name': 'Combinatorial Explorer',
                'focus': 'Novel combinations and emergent properties',
                'approach': 'Explores unconventional combinations for new functionalities'
            },
            'nanoscale_thinker': {
                'name': 'Nanoscale Architect',
                'focus': 'Nanostructures and interfaces',
                'approach': 'Focuses on nanoscale effects and interfacial engineering'
            },
            'energy_thinker': {
                'name': 'Energy Materials Specialist',
                'focus': 'Energy conversion and storage',
                'approach': 'Specializes in photovoltaic, catalytic, and energy storage materials'
            }
        }

    @cached
    def creative_formulation_generation(self, compounds: List[Dict], strategy: Dict, 
                                     innovation_factor: float) -> List[Dict]:
        """Generate formulations using creative AI thinking"""
        try:
            if not compounds:
                self.logger.warning("No compounds provided, using fallback formulations")
                return self._get_fallback_formulations(strategy)
            
            all_formulations = []
            
            # Let each creative agent generate formulations
            for agent_name, agent_info in self.creative_agents.items():
                try:
                    formulations = getattr(self, f'_{agent_name}_formulation')(
                        compounds, strategy, innovation_factor
                    )
                    all_formulations.extend(formulations)
                    
                    # Memory management
                    if MemoryManager.check_memory_limit():
                        MemoryManager.cleanup_memory()
                        
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    continue
            
            if not all_formulations:
                self.logger.warning("All agents failed, using fallback")
                return self._get_fallback_formulations(strategy)
            
            # Add computational enhancement predictions
            enhanced_formulations = []
            for formulation in all_formulations:
                try:
                    enhanced = self._enhance_with_computational_insights(formulation, strategy)
                    enhanced_formulations.append(enhanced)
                except Exception as e:
                    self.logger.error(f"Enhancement failed for formulation: {e}")
                    enhanced_formulations.append(formulation)  # Keep original
            
            return self._select_most_promising(enhanced_formulations, strategy)
            
        except Exception as e:
            self.logger.error(f"Creative formulation generation failed: {e}")
            return self._get_fallback_formulations(strategy)

    def _quantum_thinker_formulation(self, compounds: List[Dict], strategy: Dict, 
                                   innovation_factor: float) -> List[Dict]:
        """Quantum-thinking agent formulations"""
        formulations = []
        
        # Focus on electronic properties
        electronic_compounds = [c for c in compounds if self._has_electronic_character(c)]
        
        # If no electronic compounds, use all compounds
        if not electronic_compounds:
            electronic_compounds = compounds
        
        num_formulations = 3 + int(innovation_factor * 7)
        
        for _ in range(num_formulations):
            try:
                # Create formulations with electronic functionality
                num_components = random.randint(1, min(4, len(electronic_compounds)))
                selected = random.sample(electronic_compounds, num_components)
                
                formulation = self._create_creative_formulation(selected, strategy, 'quantum_thinker')
                formulation['thinking'] = "Designed for optimal electronic structure and charge transport"
                formulations.append(formulation)
            except Exception as e:
                self.logger.warning(f"Quantum thinker formulation failed: {e}")
                continue
        
        return formulations

    def _biomimetic_thinker_formulation(self, compounds: List[Dict], strategy: Dict, 
                                      innovation_factor: float) -> List[Dict]:
        """Biomimetic agent formulations"""
        formulations = []
        
        # Look for bio-inspired compounds
        bio_compounds = [c for c in compounds if self._has_biological_relevance(c)]
        
        # If no bio compounds, use all compounds
        if not bio_compounds:
            bio_compounds = compounds
        
        num_formulations = 2 + int(innovation_factor * 5)
        
        for _ in range(num_formulations):
            try:
                # Create nature-inspired formulations
                num_components = min(3, len(bio_compounds))
                selected = random.sample(bio_compounds, num_components)
                
                formulation = self._create_creative_formulation(selected, strategy, 'biomimetic_thinker')
                formulation['thinking'] = "Inspired by natural systems and biological materials"
                formulations.append(formulation)
            except Exception as e:
                self.logger.warning(f"Biomimetic thinker formulation failed: {e}")
                continue
        
        return formulations

    def _combinatorial_thinker_formulation(self, compounds: List[Dict], strategy: Dict, 
                                         innovation_factor: float) -> List[Dict]:
        """Combinatorial explorer agent formulations"""
        formulations = []
        
        num_formulations = 4 + int(innovation_factor * 8)
        
        for _ in range(num_formulations):
            try:
                # Mix different classes of compounds
                organic = [c for c in compounds if self._is_organic(c)]
                inorganic = [c for c in compounds if self._is_inorganic(c)]
                polymeric = [c for c in compounds if self._is_polymeric(c)]
                
                selected = []
                if organic: 
                    selected.append(random.choice(organic))
                if inorganic: 
                    selected.append(random.choice(inorganic))
                if polymeric and len(selected) < 3: 
                    selected.append(random.choice(polymeric))
                
                # If we don't have enough components, add random ones
                while len(selected) < min(3, len(compounds)):
                    available = [c for c in compounds if c not in selected]
                    if available:
                        selected.append(random.choice(available))
                    else:
                        break
                
                if not selected:
                    selected = random.sample(compounds, min(3, len(compounds)))
                
                formulation = self._create_creative_formulation(selected, strategy, 'combinatorial_thinker')
                formulation['thinking'] = "Exploring emergent properties from unconventional combinations"
                formulations.append(formulation)
            except Exception as e:
                self.logger.warning(f"Combinatorial thinker formulation failed: {e}")
                continue
        
        return formulations

    def _nanoscale_thinker_formulation(self, compounds: List[Dict], strategy: Dict, 
                                     innovation_factor: float) -> List[Dict]:
        """Nanoscale architect agent formulations"""
        formulations = []
        
        # Focus on compounds with nanoscale potential
        nano_compounds = [c for c in compounds if self._has_nanoscale_potential(c)]
        
        # If no nano compounds, use all compounds
        if not nano_compounds:
            nano_compounds = compounds
        
        num_formulations = 3 + int(innovation_factor * 6)
        
        for _ in range(num_formulations):
            try:
                num_components = min(4, len(nano_compounds))
                selected = random.sample(nano_compounds, num_components)
                
                formulation = self._create_creative_formulation(selected, strategy, 'nanoscale_thinker')
                formulation['thinking'] = "Designed for nanoscale effects and interfacial engineering"
                formulations.append(formulation)
            except Exception as e:
                self.logger.warning(f"Nanoscale thinker formulation failed: {e}")
                continue
        
        return formulations

    def _energy_thinker_formulation(self, compounds: List[Dict], strategy: Dict, 
                                  innovation_factor: float) -> List[Dict]:
        """Energy materials specialist agent formulations"""
        formulations = []
        
        # Focus on energy-related compounds
        energy_compounds = [c for c in compounds if self._has_energy_relevance(c)]
        
        # If no energy compounds, use all compounds
        if not energy_compounds:
            energy_compounds = compounds
        
        num_formulations = 3 + int(innovation_factor * 7)
        
        for _ in range(num_formulations):
            try:
                num_components = min(4, len(energy_compounds))
                selected = random.sample(energy_compounds, num_components)
                
                formulation = self._create_creative_formulation(selected, strategy, 'energy_thinker')
                formulation['thinking'] = "Optimized for energy conversion and storage applications"
                formulations.append(formulation)
            except Exception as e:
                self.logger.warning(f"Energy thinker formulation failed: {e}")
                continue
        
        return formulations

    def _create_creative_formulation(self, compounds: List[Dict], strategy: Dict, 
                                   agent_type: str) -> Dict[str, Any]:
        """Create a creatively designed formulation"""
        # Generate innovative mass percentages
        mass_percentages = self._generate_innovative_percentages(len(compounds))
        
        composition = []
        for i, compound in enumerate(compounds):
            comp_data = {
                'cid': compound.get('cid'),
                'name': compound.get('name', 'Unknown'),
                'mass_percentage': mass_percentages[i],
                'molecular_weight': compound.get('molecular_weight'),
                'smiles': compound.get('smiles'),
                'category': compound.get('category', 'creative')
            }
            composition.append(comp_data)
        
        # Calculate mole ratios
        composition = self._calculate_mole_ratios(composition)
        
        return {
            'composition': composition,
            'agent_type': self.creative_agents[agent_type]['name'],
            'strategy_alignment': random.uniform(0.7, 0.95),
            'confidence': random.uniform(0.6, 0.9),
            'innovation_score': random.uniform(0.5, 0.95),
            'computational_insights': {},
            'scientific_evaluation': {}
        }

    def _enhance_with_computational_insights(self, formulation: Dict, strategy: Dict) -> Dict:
        """Add computational chemistry insights to formulation"""
        try:
            # Get computational predictions
            target_props = strategy.get('target_properties', {})
            computational_insights = self.computational_predictor.predict_advanced_properties(
                formulation, target_props
            )
            
            # Get AI scientific evaluation
            scientific_eval = self.strategist.evaluate_formulation_science(formulation, strategy)
            
            formulation['computational_insights'] = computational_insights
            formulation['scientific_evaluation'] = scientific_eval
            
            return formulation
        except Exception as e:
            self.logger.error(f"Error in computational enhancement: {e}")
            return formulation

    def _select_most_promising(self, formulations: List[Dict], strategy: Dict) -> List[Dict]:
        """Select most promising formulations based on multiple criteria"""
        scored_formulations = []
        
        for formulation in formulations:
            score = self._calculate_promising_score(formulation, strategy)
            formulation['promising_score'] = score
            scored_formulations.append(formulation)
        
        # Sort by promising score
        scored_formulations.sort(key=lambda x: x['promising_score'], reverse=True)
        return scored_formulations[:20]  # Return top 20

    def _calculate_promising_score(self, formulation: Dict, strategy: Dict) -> float:
        """Calculate how promising a formulation is"""
        base_score = formulation.get('innovation_score', 0.5)
        strategy_alignment = formulation.get('strategy_alignment', 0.5)
        confidence = formulation.get('confidence', 0.5)
        
        # Consider computational insights
        comp_insights = formulation.get('computational_insights', {})
        if comp_insights:
            try:
                avg_confidence = np.mean([insight.get('confidence', 0) for insight in comp_insights.values()])
                base_score += avg_confidence * 0.2
            except:
                pass
        
        # Consider scientific evaluation
        sci_eval = formulation.get('scientific_evaluation', {})
        feasibility = sci_eval.get('scientific_feasibility', 'Medium')
        feasibility_score = {'High': 0.3, 'Medium': 0.15, 'Low': 0.0}.get(feasibility, 0.1)
        
        return (base_score * 0.4 + strategy_alignment * 0.3 + 
                confidence * 0.2 + feasibility_score)

    # Helper methods for compound classification
    def _has_electronic_character(self, compound: Dict) -> bool:
        """Check if compound has electronic functionality"""
        name = compound.get('name', '').lower()
        smiles = compound.get('smiles', '').lower()
        electronic_indicators = ['conjugated', 'aromatic', 'polymeric', 'fullerene', 
                               'graphene', 'nanotube', 'quantum', 'semiconductor']
        return any(indicator in name or indicator in smiles 
                  for indicator in electronic_indicators)

    def _has_biological_relevance(self, compound: Dict) -> bool:
        """Check if compound is biologically relevant"""
        name = compound.get('name', '').lower()
        bio_indicators = ['enzyme', 'protein', 'lipid', 'sugar', 'carbohydrate', 
                         'amino', 'peptide', 'biological', 'natural']
        return any(indicator in name for indicator in bio_indicators)

    def _is_organic(self, compound: Dict) -> bool:
        """Check if compound is organic"""
        formula = compound.get('molecular_formula', '')
        return 'C' in formula and formula.count('C') > 0

    def _is_inorganic(self, compound: Dict) -> bool:
        """Check if compound is inorganic"""
        name = compound.get('name', '').lower()
        inorganic_indicators = ['oxide', 'salt', 'mineral', 'metal', 'silicate', 
                              'zeolite', 'ceramic']
        return any(indicator in name for indicator in inorganic_indicators)

    def _is_polymeric(self, compound: Dict) -> bool:
        """Check if compound is polymeric"""
        name = compound.get('name', '').lower()
        mw = compound.get('molecular_weight', 0)
        return 'poly' in name or mw > 1000

    def _has_nanoscale_potential(self, compound: Dict) -> bool:
        """Check if compound has nanoscale potential"""
        name = compound.get('name', '').lower()
        nano_indicators = ['nano', 'quantum', 'cluster', 'particle', 'dot', 
                          'tube', 'sheet', 'wire']
        return any(indicator in name for indicator in nano_indicators)

    def _has_energy_relevance(self, compound: Dict) -> bool:
        """Check if compound is energy-relevant"""
        name = compound.get('name', '').lower()
        energy_indicators = ['photo', 'solar', 'battery', 'fuel', 'catalyst', 
                           'electrode', 'conductor', 'storage']
        return any(indicator in name for indicator in energy_indicators)

    def _generate_innovative_percentages(self, num_components: int) -> List[float]:
        """Generate innovative mass percentages"""
        if num_components == 1:
            return [100.0]
        
        # More creative distribution than simple random
        if num_components == 2:
            # Often one major component
            major = random.uniform(60, 90)
            return [major, 100 - major]
        elif num_components == 3:
            # Various distributions
            distribution_type = random.choice(['balanced', 'major_minor', 'gradient'])
            if distribution_type == 'balanced':
                base = 100 / num_components
                return [base * random.uniform(0.7, 1.3) for _ in range(num_components)]
            elif distribution_type == 'major_minor':
                major = random.uniform(50, 80)
                remaining = 100 - major
                minor1 = remaining * random.uniform(0.3, 0.7)
                return [major, minor1, remaining - minor1]
            else:  # gradient
                return sorted([random.uniform(5, 60) for _ in range(num_components)], reverse=True)
        else:
            # Complex distributions
            percentages = [random.uniform(1, 40) for _ in range(num_components)]
            total = sum(percentages)
            return [p/total * 100 for p in percentages]

    def _calculate_mole_ratios(self, composition: List[Dict]) -> List[Dict]:
        """Calculate mole ratios and percentages"""
        total_moles = 0
        mole_data = []
        
        for comp in composition:
            mass_percent = comp['mass_percentage']
            mw = comp.get('molecular_weight')
            
            if mw and mw > 0:
                moles = mass_percent / mw
            else:
                moles = mass_percent / 100  # Fallback
                
            mole_data.append(moles)
            total_moles += moles
        
        if total_moles > 0:
            for i, comp in enumerate(composition):
                comp['mole_percentage'] = (mole_data[i] / total_moles) * 100
                comp['mole_ratio'] = mole_data[i] / min(mole_data) if min(mole_data) > 0 else 1
        else:
            for comp in composition:
                comp['mole_percentage'] = 100.0 / len(composition)
                comp['mole_ratio'] = 1.0
        
        return composition

    def _get_fallback_formulations(self, strategy: Dict) -> List[Dict]:
        """Provide basic formulations when generation fails"""
        self.logger.info("Using fallback formulations")
        return [{
            'composition': [{
                'name': 'Basic Formulation', 
                'mass_percentage': 100.0,
                'category': 'fallback'
            }],
            'agent_type': 'Fallback Agent',
            'strategy_alignment': 0.5,
            'confidence': 0.3,
            'innovation_score': 0.3,
            'computational_insights': {},
            'scientific_evaluation': {'scientific_feasibility': 'Medium'},
            'thinking': 'Basic fallback formulation'
        }]

# Fallback classes for when dependencies are missing
class AIStrategistFallback:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def think_about_challenge(self, challenge_text: str, material_type: str) -> Dict[str, Any]:
        return {
            "scientific_analysis": f"Basic analysis for {material_type}",
            "key_mechanisms": ["basic functionality"],
            "target_compound_classes": [material_type],
            "search_strategy": {
                "primary_terms": [material_type],
                "innovative_terms": []
            }
        }
    
    def evaluate_formulation_science(self, formulation: Dict, strategy: Dict) -> Dict[str, Any]:
        return {
            "scientific_feasibility": "Medium",
            "key_advantages": ["Basic functionality"],
            "potential_issues": []
        }

class ComputationalPredictorFallback:
    def predict_advanced_properties(self, formulation: Dict, target_props: Dict) -> Dict[str, Any]:
        return {}
