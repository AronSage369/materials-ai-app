import random
from typing import Dict, List, Any, Optional
import numpy as np
import logging

class CreativeAIEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.creative_agents = self._initialize_innovative_agents()
        
    def _initialize_innovative_agents(self) -> Dict[str, Any]:
        """Initialize highly innovative AI agents"""
        return {
            'quantum_architect': {
                'name': 'Quantum Materials Architect',
                'focus': 'Quantum effects, electronic structure, nanoscale phenomena',
                'approach': 'Designs materials with quantum confinement, band engineering, and emergent electronic properties',
                'innovation_level': 0.9
            },
            'bio_inspired_designer': {
                'name': 'Bio-Inspired Materials Designer', 
                'focus': 'Nature-inspired solutions, self-assembly, biological principles',
                'approach': 'Leverages evolutionary optimized biological systems and biomimetic principles',
                'innovation_level': 0.8
            },
            'multiscale_engineer': {
                'name': 'Multiscale Systems Engineer',
                'focus': 'Hierarchical structures, interface engineering, composite systems',
                'approach': 'Engineers materials across multiple length scales for optimal performance',
                'innovation_level': 0.85
            },
            'emergent_properties_explorer': {
                'name': 'Emergent Properties Explorer',
                'focus': 'Synergistic combinations, unexpected functionalities, novel phenomena',
                'approach': 'Seeks formulations where components interact to create new capabilities',
                'innovation_level': 0.95
            },
            'computational_alchemist': {
                'name': 'Computational Alchemist',
                'focus': 'High-risk high-reward combinations, unconventional mixtures',
                'approach': 'Explores chemical space boundaries and unconventional formulations',
                'innovation_level': 1.0
            }
        }

    def creative_formulation_generation(self, compounds: List[Dict], strategy: Dict, 
                                     innovation_factor: float) -> List[Dict]:
        """Generate highly innovative formulations using advanced AI thinking"""
        all_formulations = []
        
        # Each agent generates formulations with their unique perspective
        for agent_name, agent_info in self.creative_agents.items():
            try:
                formulations = getattr(self, f'_{agent_name}_formulation')(
                    compounds, strategy, innovation_factor
                )
                all_formulations.extend(formulations)
                self.logger.info(f"Agent {agent_name} generated {len(formulations)} formulations")
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                continue
        
        # Ensure minimum formulations
        if len(all_formulations) < 10:
            additional = self._emergency_creative_generation(compounds, strategy, 10 - len(all_formulations))
            all_formulations.extend(additional)
        
        # Score and select most promising
        scored_formulations = []
        for formulation in all_formulations:
            score = self._calculate_innovation_score(formulation, strategy, innovation_factor)
            formulation['innovation_score'] = score
            scored_formulations.append(formulation)
        
        # Sort by innovation score
        scored_formulations.sort(key=lambda x: x['innovation_score'], reverse=True)
        return scored_formulations[:25]  # Return top 25 most innovative

    def _quantum_architect_formulation(self, compounds: List[Dict], strategy: Dict, 
                                    innovation_factor: float) -> List[Dict]:
        """Quantum architect formulations focusing on electronic properties"""
        formulations = []
        
        # Find compounds with quantum-relevant properties
        quantum_compounds = [c for c in compounds if self._is_quantum_relevant(c)]
        
        num_formulations = 5 + int(innovation_factor * 10)
        
        for _ in range(num_formulations):
            try:
                # Create quantum-optimized formulations
                num_components = random.randint(2, 5)
                if quantum_compounds:
                    selected = random.sample(quantum_compounds, min(num_components, len(quantum_compounds)))
                else:
                    selected = random.sample(compounds, min(num_components, len(compounds)))
                
                formulation = self._create_innovative_formulation(selected, strategy, 'quantum_architect')
                formulation['thinking'] = "Quantum-optimized for electronic structure and nanoscale effects"
                formulations.append(formulation)
            except Exception as e:
                continue
        
        return formulations

    def _bio_inspired_designer_formulation(self, compounds: List[Dict], strategy: Dict, 
                                         innovation_factor: float) -> List[Dict]:
        """Bio-inspired formulations"""
        formulations = []
        
        # Find bio-relevant compounds
        bio_compounds = [c for c in compounds if self._is_bio_relevant(c)]
        
        num_formulations = 4 + int(innovation_factor * 8)
        
        for _ in range(num_formulations):
            try:
                # Create nature-inspired formulations
                num_components = random.randint(2, 4)
                if bio_compounds:
                    selected = random.sample(bio_compounds, min(num_components, len(bio_compounds)))
                else:
                    selected = random.sample(compounds, min(num_components, len(compounds)))
                
                formulation = self._create_innovative_formulation(selected, strategy, 'bio_inspired_designer')
                formulation['thinking'] = "Bio-inspired design leveraging natural principles and self-assembly"
                formulations.append(formulation)
            except Exception as e:
                continue
        
        return formulations

    def _multiscale_engineer_formulation(self, compounds: List[Dict], strategy: Dict, 
                                       innovation_factor: float) -> List[Dict]:
        """Multiscale engineered formulations"""
        formulations = []
        
        # Categorize compounds by scale
        molecular_scale = [c for c in compounds if self._is_molecular_scale(c)]
        nano_scale = [c for c in compounds if self._is_nano_scale(c)]
        macro_scale = [c for c in compounds if self._is_macro_scale(c)]
        
        num_formulations = 6 + int(innovation_factor * 12)
        
        for _ in range(num_formulations):
            try:
                # Mix across scales
                selected = []
                if molecular_scale: selected.append(random.choice(molecular_scale))
                if nano_scale: selected.append(random.choice(nano_scale))
                if macro_scale and len(selected) < 3: selected.append(random.choice(macro_scale))
                
                # Fill remaining slots
                while len(selected) < min(4, len(compounds)):
                    available = [c for c in compounds if c not in selected]
                    if available:
                        selected.append(random.choice(available))
                    else:
                        break
                
                formulation = self._create_innovative_formulation(selected, strategy, 'multiscale_engineer')
                formulation['thinking'] = "Multiscale engineered for hierarchical structure and optimized interfaces"
                formulations.append(formulation)
            except Exception as e:
                continue
        
        return formulations

    def _emergent_properties_explorer_formulation(self, compounds: List[Dict], strategy: Dict, 
                                                innovation_factor: float) -> List[Dict]:
        """Formulations targeting emergent properties"""
        formulations = []
        
        num_formulations = 8 + int(innovation_factor * 15)
        
        for _ in range(num_formulations):
            try:
                # Create unconventional combinations for emergent properties
                num_components = random.randint(3, 6)
                
                # Ensure chemical diversity
                selected = []
                categories_used = set()
                
                while len(selected) < num_components and len(selected) < len(compounds):
                    available = [c for c in compounds if c not in selected]
                    if not available:
                        break
                    
                    # Prefer compounds from unused categories
                    candidate = random.choice(available)
                    cat = self._categorize_compound(candidate)
                    if cat not in categories_used:
                        selected.append(candidate)
                        categories_used.add(cat)
                    else:
                        # Sometimes add from used category for complexity
                        if random.random() < 0.3:
                            selected.append(candidate)
                
                formulation = self._create_innovative_formulation(selected, strategy, 'emergent_properties_explorer')
                formulation['thinking'] = "Designed for emergent properties and synergistic interactions"
                formulations.append(formulation)
            except Exception as e:
                continue
        
        return formulations

    def _computational_alchemist_formulation(self, compounds: List[Dict], strategy: Dict, 
                                           innovation_factor: float) -> List[Dict]:
        """High-risk high-reward formulations"""
        formulations = []
        
        num_formulations = 3 + int(innovation_factor * 7)
        
        for _ in range(num_formulations):
            try:
                # Create truly unconventional formulations
                num_components = random.randint(2, 8)
                
                # Use weighted random selection based on molecular complexity
                weights = [c.get('complexity', 1) for c in compounds]
                if sum(weights) > 0:
                    selected = random.choices(compounds, weights=weights, k=min(num_components, len(compounds)))
                else:
                    selected = random.sample(compounds, min(num_components, len(compounds)))
                
                formulation = self._create_innovative_formulation(selected, strategy, 'computational_alchemist')
                formulation['thinking'] = "High-risk exploration of unconventional chemical space"
                formulation['risk_level'] = random.uniform(0.7, 1.0)
                formulations.append(formulation)
            except Exception as e:
                continue
        
        return formulations

    def _create_innovative_formulation(self, compounds: List[Dict], strategy: Dict, 
                                     agent_type: str) -> Dict[str, Any]:
        """Create an innovative formulation"""
        # Use innovative mass distribution strategies
        mass_percentages = self._innovative_mass_distribution(len(compounds))
        
        composition = []
        for i, compound in enumerate(compounds):
            comp_data = {
                'cid': compound.get('cid'),
                'name': compound.get('name', 'Unknown'),
                'mass_percentage': mass_percentages[i],
                'molecular_weight': compound.get('molecular_weight'),
                'smiles': compound.get('smiles'),
                'category': compound.get('category', 'innovative'),
                'complexity': compound.get('complexity', 1)
            }
            composition.append(comp_data)
        
        # Calculate advanced properties
        composition = self._calculate_advanced_ratios(composition)
        
        return {
            'composition': composition,
            'agent_type': self.creative_agents[agent_type]['name'],
            'agent_innovation_level': self.creative_agents[agent_type]['innovation_level'],
            'strategy_alignment': random.uniform(0.6, 0.95),
            'chemical_diversity': self._calculate_diversity(composition),
            'complexity_score': self._calculate_complexity(composition),
            'synergy_potential': random.uniform(0.5, 0.9)
        }

    def _innovative_mass_distribution(self, num_components: int) -> List[float]:
        """Generate innovative mass distributions"""
        if num_components == 1:
            return [100.0]
        
        distributions = [
            # Major-minor distribution
            lambda n: [70] + [30/(n-1)] * (n-1) if n > 1 else [100],
            # Balanced distribution
            lambda n: [100/n] * n,
            # Gradient distribution
            lambda n: sorted([random.uniform(5, 60) for _ in range(n)], reverse=True),
            # Bimodal distribution
            lambda n: [40, 40] + [20/(n-2)] * (n-2) if n > 2 else [60, 40],
            # Random complex distribution
            lambda n: [random.uniform(1, 80) for _ in range(n)]
        ]
        
        # Choose random distribution and normalize
        distribution = random.choice(distributions)(num_components)
        total = sum(distribution)
        return [d/total * 100 for d in distribution]

    def _calculate_innovation_score(self, formulation: Dict, strategy: Dict, innovation_factor: float) -> float:
        """Calculate innovation score considering multiple factors"""
        base_innovation = formulation.get('agent_innovation_level', 0.5)
        diversity = formulation.get('chemical_diversity', 0.5)
        complexity = formulation.get('complexity_score', 0.5)
        synergy = formulation.get('synergy_potential', 0.5)
        alignment = formulation.get('strategy_alignment', 0.5)
        
        # Weight factors based on innovation factor
        weights = {
            'base_innovation': 0.3,
            'diversity': 0.2,
            'complexity': 0.15,
            'synergy': 0.2,
            'alignment': 0.15
        }
        
        # Adjust weights for higher innovation factor
        if innovation_factor > 0.7:
            weights.update({'base_innovation': 0.4, 'complexity': 0.25, 'alignment': 0.1})
        
        score = (base_innovation * weights['base_innovation'] +
                diversity * weights['diversity'] +
                complexity * weights['complexity'] +
                synergy * weights['synergy'] +
                alignment * weights['alignment'])
        
        return min(1.0, score * (1 + innovation_factor * 0.5))

    # Enhanced compound classification methods
    def _is_quantum_relevant(self, compound: Dict) -> bool:
        name = compound.get('name', '').lower()
        smiles = compound.get('smiles', '').lower()
        quantum_indicators = ['quantum', 'nanotube', 'graphene', '2D', 'semiconductor', 
                            'conjugated', 'aromatic', 'fullerene', 'perovskite']
        return any(indicator in name or indicator in smiles for indicator in quantum_indicators)

    def _is_bio_relevant(self, compound: Dict) -> bool:
        name = compound.get('name', '').lower()
        bio_indicators = ['enzyme', 'protein', 'lipid', 'sugar', 'amino', 'peptide',
                         'biological', 'natural', 'plant', 'animal', 'cell']
        return any(indicator in name for indicator in bio_indicators)

    def _is_molecular_scale(self, compound: Dict) -> bool:
        mw = compound.get('molecular_weight', 0)
        return mw < 500

    def _is_nano_scale(self, compound: Dict) -> bool:
        name = compound.get('name', '').lower()
        mw = compound.get('molecular_weight', 0)
        return 'nano' in name or mw > 1000

    def _is_macro_scale(self, compound: Dict) -> bool:
        mw = compound.get('molecular_weight', 0)
        return mw > 5000

    def _categorize_compound(self, compound: Dict) -> str:
        """Categorize compound for diversity calculation"""
        name = compound.get('name', '').lower()
        if any(word in name for word in ['polymer', 'poly']):
            return 'polymer'
        elif any(word in name for word in ['nanotube', 'graphene', 'quantum']):
            return 'nanomaterial'
        elif any(word in name for word in ['acid', 'base', 'salt']):
            return 'inorganic'
        elif any(word in name for word in ['organic', 'carbon']):
            return 'organic'
        elif any(word in name for word in ['metal', 'oxide']):
            return 'metal'
        else:
            return 'other'

    def _calculate_diversity(self, composition: List[Dict]) -> float:
        """Calculate chemical diversity of formulation"""
        categories = [self._categorize_compound(comp) for comp in composition]
        unique_categories = len(set(categories))
        max_possible = min(len(composition), 5)  # Maximum of 5 categories
        return unique_categories / max_possible

    def _calculate_complexity(self, composition: List[Dict]) -> float:
        """Calculate molecular complexity of formulation"""
        complexities = [comp.get('complexity', 1) for comp in composition]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            return min(1.0, avg_complexity / 500)  # Normalize
        return 0.5

    def _calculate_advanced_ratios(self, composition: List[Dict]) -> List[Dict]:
        """Calculate advanced molecular ratios"""
        total_mass = sum(comp['mass_percentage'] for comp in composition)
        
        for comp in composition:
            mass_frac = comp['mass_percentage'] / total_mass
            mw = comp.get('molecular_weight', 100)
            
            # Calculate mole-based properties
            comp['mole_fraction'] = mass_frac / mw if mw > 0 else mass_frac
            comp['volume_estimate'] = mass_frac * mw / 1000  # Rough estimate
            
        # Normalize mole fractions
        total_moles = sum(comp['mole_fraction'] for comp in composition)
        if total_moles > 0:
            for comp in composition:
                comp['mole_fraction'] /= total_moles
        
        return composition

    def _emergency_creative_generation(self, compounds: List[Dict], strategy: Dict, 
                                     num_needed: int) -> List[Dict]:
        """Emergency formulation generation when other methods fail"""
        formulations = []
        
        for _ in range(num_needed):
            try:
                num_components = random.randint(1, min(4, len(compounds)))
                selected = random.sample(compounds, num_components)
                
                formulation = self._create_innovative_formulation(selected, strategy, 'computational_alchemist')
                formulation['thinking'] = "Emergency creative formulation"
                formulation['emergency_generated'] = True
                formulations.append(formulation)
            except:
                continue
        
        return formulations
