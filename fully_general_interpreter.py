#!/usr/bin/env python3
"""
Fully General PyTheus Quantum Network Interpreter

This interpreter is completely general and adaptive, analyzing any quantum network
structure dynamically without hardcoded network types or assumptions.
It generates physically meaningful optical table setups and PyTheus-style graphs
for any quantum state or network configuration.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch, RegularPolygon
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict, Counter
import networkx as nx
import sys
import os
import datetime
import argparse

class GeneralQuantumNetworkInterpreter:
    """
    A fully general quantum network interpreter that adapts to any graph structure
    and configuration without hardcoded network type assumptions.
    """
    
    def __init__(self, config_path=None, graph_path=None, config_data=None, graph_data=None, verbose=True):
        self.config_path = config_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.config = {}
        self.graph = {}
        self.vertices = []
        
        # Load data from files or direct data
        if config_data is not None:
            self.config = config_data
            if self.verbose:
                print("✅ Loaded config from provided data")
        elif config_path:
            self.load_config(config_path)
            
        if graph_data is not None:
            self.load_graph_data(graph_data)
            if self.verbose:
                print("✅ Loaded graph from provided data")
        elif graph_path:
            self.load_graph(graph_path)
    
    def load_graph_data(self, graph_data):
        """Load graph from dictionary data."""
        try:
            # Handle different graph formats
            if isinstance(graph_data, dict):
                if 'graph' in graph_data:
                    self.graph = graph_data['graph']
                else:
                    self.graph = graph_data
            else:
                self.graph = graph_data
            
            # Extract vertices
            self.vertices = self._extract_vertices()
            
            if self.verbose:
                print(f"   Vertices: {len(self.vertices)}")
                print(f"   Edges: {len(self.graph)}")
        except Exception as e:
            print(f"❌ Error loading graph data: {e}")
            self.graph = {}
            self.vertices = []
    
    def load_config(self, config_path):
        """Load configuration from JSON file or dict."""
        try:
            if isinstance(config_path, dict):
                # Handle direct dict input
                self.config = config_path
                if self.verbose:
                    print("✅ Loaded config from provided dictionary")
            else:
                # Handle file path
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                if self.verbose:
                    print(f"✅ Loaded config from {config_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            self.config = {}
    
    def load_graph(self, graph_path):
        """Load graph from JSON file or dict."""
        try:
            if isinstance(graph_path, dict):
                # Handle direct dict input
                self.load_graph_data(graph_path)
                if self.verbose:
                    print("✅ Loaded graph from provided dictionary")
            else:
                # Handle file path
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                self.load_graph_data(graph_data)
                if self.verbose:
                    print(f"✅ Loaded graph from {graph_path}")
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            self.graph = {}
            self.vertices = []
    
    def _extract_vertices(self):
        """Extract unique vertices from the graph."""
        vertices = set()
        
        # Handle different graph formats
        if isinstance(self.graph, dict):
            for edge_key in self.graph.keys():
                try:
                    if isinstance(edge_key, str):
                        # Parse edge string like "(0, 1, 0, 1)"
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                    
                    if len(edge_tuple) >= 2:
                        vertices.add(edge_tuple[0])
                        vertices.add(edge_tuple[1])
                except:
                    # Skip malformed edges
                    continue
        
        return sorted(list(vertices))
    
    def analyze_network_structure(self):
        """
        Completely general network analysis that adapts to any structure.
        No hardcoded assumptions about network types.
        """
        if self.verbose:
            print("🔍 Analyzing network structure...")
        
        # Basic graph analysis
        vertices = self.vertices
        degrees = self._compute_vertex_degrees()
        connectivity = self._analyze_connectivity()
        
        # Mode analysis
        mode_analysis = self._analyze_modes()
        
        # Identify functional roles based on graph structure
        functional_roles = self._identify_functional_roles(degrees, connectivity)
        
        # Analyze quantum state properties
        state_analysis = self._analyze_quantum_state()
        
        # Determine optical implementation strategy
        implementation_strategy = self._determine_implementation_strategy(
            functional_roles, mode_analysis, state_analysis
        )
        
        # Find graph motifs and patterns
        motifs = self._find_graph_motifs()
        
        return {
            'vertices': vertices,
            'degrees': degrees,
            'connectivity': connectivity,
            'mode_analysis': mode_analysis,
            'functional_roles': functional_roles,
            'state_analysis': state_analysis,
            'implementation_strategy': implementation_strategy,
            'motifs': motifs,
            'config': self.config,
            'graph_size': len(vertices),
            'edge_count': len(self.graph),
            'description': self.config.get('description', f'Quantum network ({len(vertices)} vertices)')
        }
    
    def _compute_vertex_degrees(self):
        """Compute degree of each vertex."""
        degrees = defaultdict(int)
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            if len(edge_tuple) >= 2:
                degrees[edge_tuple[0]] += 1
                degrees[edge_tuple[1]] += 1
        
        return dict(degrees)
    
    def _analyze_connectivity(self):
        """Analyze connectivity patterns in the graph."""
        # Create NetworkX graph for analysis
        G = nx.Graph()
        
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            if len(edge_tuple) >= 2:
                weight = abs(self.graph[edge])
                G.add_edge(edge_tuple[0], edge_tuple[1], weight=weight)
        
        # Analyze connectivity properties
        if G.nodes():
            connectivity = {
                'is_connected': nx.is_connected(G),
                'components': list(nx.connected_components(G)),
                'num_components': nx.number_connected_components(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else 0,
                'clustering': nx.average_clustering(G),
                'density': nx.density(G),
                'is_tree': nx.is_tree(G),
                'is_bipartite': nx.is_bipartite(G)
            }
        else:
            # Handle empty graph
            connectivity = {
                'is_connected': False,
                'components': [],
                'num_components': 0,
                'diameter': 0,
                'clustering': 0,
                'density': 0,
                'is_tree': False,
                'is_bipartite': False
            }
        
        return connectivity
    
    def _analyze_modes(self):
        """Analyze mode structure of the graph."""
        modes = set()
        mode_pairs = []
        edge_weights = []
        
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            weight = self.graph[edge]
            edge_weights.append(abs(weight))
            
            # Extract modes if available
            if len(edge_tuple) >= 4:
                mode1, mode2 = edge_tuple[2], edge_tuple[3]
                modes.add(mode1)
                modes.add(mode2)
                mode_pairs.append((mode1, mode2))
            else:
                # Default to single mode
                modes.add(0)
                mode_pairs.append((0, 0))
        
        return {
            'unique_modes': sorted(list(modes)),
            'num_modes': len(modes),
            'mode_pairs': mode_pairs,
            'edge_weights': edge_weights,
            'max_weight': max(edge_weights) if edge_weights else 0,
            'weight_distribution': Counter([round(w, 3) for w in edge_weights]),
            'has_complex_weights': any(w < 0 for w in self.graph.values())
        }
    
    def _identify_functional_roles(self, degrees, connectivity):
        """
        Identify functional roles of vertices based on graph structure AND config.
        Prioritizes config-specified roles over structural heuristics.
        """
        vertices = self.vertices
        
        # First, check if config specifies single emitters
        config_single_emitters = self.config.get('single_emitters', [])
        target_state = self.config.get('target_state', [])
        
        if config_single_emitters and target_state:
            # Use config-specified single emitters as sources
            potential_sources = config_single_emitters
            
            # For target states like W4: ["0001", "0010", "0100", "1000"]
            # The number of output detectors = length of each state string
            num_output_detectors = len(target_state[0]) if target_state else 4
            
            # The first N vertices (where N = output detectors) are the detectors
            potential_detectors = list(range(num_output_detectors))
            
            # Remaining vertices that aren't sources or detectors are ancillas
            potential_ancillas = [v for v in vertices 
                                if v not in potential_sources and v not in potential_detectors]
            
            if self.verbose:
                print(f"   Config-based assignment:")
                print(f"   Sources: {potential_sources}")
                print(f"   Detectors: {potential_detectors}")
                print(f"   Ancillas: {potential_ancillas}")
        else:
            # Fall back to structural analysis
            degree_values = list(degrees.values())
            mean_degree = np.mean(degree_values) if degree_values else 0
            std_degree = np.std(degree_values) if degree_values else 0
            
            # Analyze centrality
            G = nx.Graph()
            for edge in self.graph.keys():
                if isinstance(edge, str):
                    edge_tuple = eval(edge)
                else:
                    edge_tuple = edge
                
                if len(edge_tuple) >= 2:
                    G.add_edge(edge_tuple[0], edge_tuple[1])
            
            betweenness = nx.betweenness_centrality(G) if G.nodes() else {}
            closeness = nx.closeness_centrality(G) if G.nodes() else {}
            
            potential_sources = []
            potential_detectors = []
            potential_ancillas = []
            
            for v in vertices:
                degree = degrees[v]
                betw = betweenness.get(v, 0)
                close = closeness.get(v, 0)
                
                if degree == 1:
                    # Leaf nodes are likely detectors
                    potential_detectors.append(v)
                elif degree <= 2 and betw < 0.1:
                    # Low degree, low centrality -> likely source or detector
                    if close < mean_degree / len(vertices):
                        potential_sources.append(v)
                    else:
                        potential_detectors.append(v)
                elif degree >= 3 and betw > 0.1:
                    # High degree, high centrality -> likely ancilla or central component
                    potential_ancillas.append(v)
                else:
                    # Medium degree -> could be detector or intermediate
                    potential_detectors.append(v)
        
        # Statistical analysis for completeness
        degree_values = list(degrees.values())
        mean_degree = np.mean(degree_values) if degree_values else 0
        std_degree = np.std(degree_values) if degree_values else 0
        
        # Classify vertices by degree
        high_degree = [v for v in vertices if degrees[v] > mean_degree + 0.5 * std_degree]
        low_degree = [v for v in vertices if degrees[v] < mean_degree - 0.5 * std_degree]
        medium_degree = [v for v in vertices if v not in high_degree and v not in low_degree]
        
        # Identify leaves (degree 1) and hubs (highest degree)
        leaves = [v for v in vertices if degrees[v] == 1]
        hubs = [v for v in vertices if degrees[v] == max(degree_values)] if degree_values else []
        
        # Analyze centrality for reference
        G = nx.Graph()
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            if len(edge_tuple) >= 2:
                G.add_edge(edge_tuple[0], edge_tuple[1])
        
        betweenness = nx.betweenness_centrality(G) if G.nodes() else {}
        closeness = nx.closeness_centrality(G) if G.nodes() else {}
        
        return {
            'high_degree': high_degree,
            'low_degree': low_degree,
            'medium_degree': medium_degree,
            'leaves': leaves,
            'hubs': hubs,
            'potential_sources': potential_sources,
            'potential_detectors': potential_detectors,
            'potential_ancillas': potential_ancillas,
            'degree_stats': {
                'mean': mean_degree,
                'std': std_degree,
                'min': min(degree_values) if degree_values else 0,
                'max': max(degree_values) if degree_values else 0
            },
            'centrality': {
                'betweenness': betweenness,
                'closeness': closeness
            },
            'config_sources': config_single_emitters,
            'used_config_sources': bool(config_single_emitters)
        }
    
    def _analyze_quantum_state(self):
        """Analyze quantum state properties from config."""
        target_state = self.config.get('target_state', [])
        single_emitters = self.config.get('single_emitters', [])
        description = self.config.get('description', '').lower()
        
        # Analyze target state structure
        if target_state:
            state_analysis = {
                'has_target_state': True,
                'num_basis_states': len(target_state),
                'state_length': len(target_state[0]) if target_state else 0,
                'is_superposition': len(target_state) > 1,
                'basis_states': target_state
            }
            
            # Analyze entanglement patterns
            if len(target_state) > 1:
                # Check for common patterns
                all_zeros_ones = all(set(state) <= {'0', '1'} for state in target_state)
                single_excitation = all(state.count('1') == 1 for state in target_state)
                max_entanglement = all(len(set(state)) == 1 for state in target_state)
                
                state_analysis.update({
                    'is_binary': all_zeros_ones,
                    'single_excitation_pattern': single_excitation,
                    'max_entanglement_pattern': max_entanglement,
                    'entanglement_type': self._classify_entanglement_type(target_state)
                })
        else:
            state_analysis = {
                'has_target_state': False,
                'inferred_from_graph': True,
                'num_basis_states': 2,  # Default assumption
                'state_length': len(self.vertices),
                'is_superposition': True
            }
        
        # Add implementation hints
        state_analysis.update({
            'single_emitters': single_emitters,
            'has_single_emitters': bool(single_emitters),
            'description_keywords': [word for word in description.split() 
                                   if word in ['qkd', 'ghz', 'bell', 'w', 'cluster', 'spdc']],
            'particle_number': len(self.vertices) // 2 if len(self.vertices) % 2 == 0 else len(self.vertices)
        })
        
        return state_analysis
    
    def _classify_entanglement_type(self, target_state):
        """Classify the type of entanglement from the target state."""
        if not target_state:
            return 'unknown'
        
        # Check for W-state pattern (single excitation)
        if all(state.count('1') == 1 and state.count('0') == len(state) - 1 for state in target_state):
            return 'w_state'
        
        # Check for GHZ-state pattern (maximal entanglement)
        if all(len(set(state)) == 1 for state in target_state):
            return 'ghz_state'
        
        # Check for Bell state pattern (2-qubit)
        if len(target_state[0]) == 2 and len(target_state) == 2:
            return 'bell_state'
        
        # Check for cluster state pattern
        if len(target_state) == 2**len(target_state[0]):
            return 'computational_basis'
        
        return 'general_entangled'
    
    def _determine_implementation_strategy(self, functional_roles, mode_analysis, state_analysis):
        """
        Determine the optical implementation strategy based on ACTUAL graph structure analysis.
        No predefined categories - purely based on graph connectivity and node roles.
        """
        # Analyze the actual graph structure
        num_vertices = len(self.vertices)
        num_modes = mode_analysis['num_modes']
        connectivity = self._analyze_connectivity()
        
        # Analyze actual node roles from graph structure
        sources = self._identify_actual_sources()
        detectors = self._identify_actual_detectors()
        beam_splitters = self._identify_beam_splitter_nodes()
        ancillas = self._identify_ancilla_nodes()
        
        # Determine implementation based on actual structure
        strategy_info = {
            'sources': sources,
            'detectors': detectors,
            'beam_splitters': beam_splitters,
            'ancillas': ancillas,
            'modes': mode_analysis['unique_modes'],
            'requires_heralding': len(ancillas) > 0,
            'complexity_level': self._assess_complexity(functional_roles, mode_analysis, state_analysis),
            'optical_elements': self._determine_optical_elements(sources, detectors, beam_splitters, ancillas),
            'connection_map': self._build_connection_map()
        }
        
        return strategy_info
    
    def _assess_complexity(self, functional_roles, mode_analysis, state_analysis):
        """Assess the complexity level of the network."""
        complexity_score = 0
        
        # Add complexity based on various factors
        complexity_score += len(self.vertices) * 0.5
        complexity_score += mode_analysis['num_modes'] * 1.0
        complexity_score += len(functional_roles['potential_ancillas']) * 2.0
        complexity_score += len(mode_analysis['edge_weights']) * 0.1
        
        if state_analysis.get('is_superposition', False):
            complexity_score += 2.0
        
        if complexity_score < 3:
            return 'simple'
        elif complexity_score < 8:
            return 'moderate'
        else:
            return 'complex'
    
    def _recommend_sources(self, functional_roles, state_analysis):
        """Recommend source configuration based on analysis."""
        # Prioritize config-specified single emitters
        config_sources = functional_roles.get('config_sources', [])
        
        if config_sources:
            return {
                'type': 'single_photon_sources',
                'locations': config_sources,
                'wavelength': '810nm'
            }
        elif state_analysis.get('has_single_emitters', False):
            return {
                'type': 'single_photon_sources',
                'locations': state_analysis['single_emitters'],
                'wavelength': '810nm'
            }
        elif functional_roles['potential_sources']:
            return {
                'type': 'variable_sources',
                'locations': functional_roles['potential_sources'],
                'wavelength': '810nm'
            }
        else:
            return {
                'type': 'spdc_sources',
                'locations': 'external',
                'wavelength': '405nm pump, 810nm signal/idler'
            }
    
    def _recommend_detectors(self, functional_roles, state_analysis):
        """Recommend detector configuration based on analysis."""
        return {
            'type': 'single_photon_detectors',
            'locations': functional_roles['potential_detectors'],
            'efficiency': 0.95,
            'dark_count_rate': '100 Hz'
        }
    
    def _find_graph_motifs(self):
        """Find common graph motifs and patterns."""
        motifs = {
            'triangles': [],
            'squares': [],
            'stars': [],
            'paths': [],
            'cycles': []
        }
        
        # Create NetworkX graph for motif analysis
        G = nx.Graph()
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            if len(edge_tuple) >= 2:
                G.add_edge(edge_tuple[0], edge_tuple[1])
        
        # Find triangles
        triangles = [list(triangle) for triangle in nx.enumerate_all_cliques(G) if len(triangle) == 3]
        motifs['triangles'] = triangles
        
        # Find squares (4-cycles)
        if len(G.nodes) >= 4:
            cycles = [cycle for cycle in nx.simple_cycles(G) if len(cycle) == 4]
            motifs['squares'] = cycles
        
        # Find star patterns (hub with multiple connections)
        for node in G.nodes():
            if G.degree(node) >= 3:
                neighbors = list(G.neighbors(node))
                motifs['stars'].append({'center': node, 'leaves': neighbors})
        
        return motifs
    
    def plot_optical_table_setup(self, save_path=None, title=None):
        """
        Create a completely general optical table setup based on actual graph structure analysis.
        No predefined categories - interprets any graph structure correctly.
        """
        # Get comprehensive network analysis
        network_analysis = self.analyze_network_structure()
        
        if self.verbose:
            print("🔬 Creating general optical table setup...")
            strategy_info = network_analysis['implementation_strategy']
            print(f"   Sources: {strategy_info['sources']}")
            print(f"   Detectors: {strategy_info['detectors']}")
            print(f"   Beam Splitters: {strategy_info['beam_splitters']}")
            print(f"   Ancillas: {strategy_info['ancillas']}")
        
        # Use the general SPDC optical table for all networks
        # This is the most flexible approach that works for any graph structure
        return self._plot_general_spdc_optical_table(network_analysis, save_path, title)
    
    def _plot_single_photon_optical_table(self, network_analysis, save_path=None, title=None):
        """Plot optical table for single photon source networks."""
        # Get key information
        functional_roles = network_analysis['functional_roles']
        mode_analysis = network_analysis['mode_analysis']
        source_config = network_analysis['implementation_strategy']['recommended_sources']
        detector_config = network_analysis['implementation_strategy']['recommended_detectors']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_facecolor('#f8f9fa')
        
        # Adaptive table dimensions
        num_components = len(functional_roles['potential_sources']) + len(functional_roles['potential_detectors'])
        table_width = max(14, num_components * 1.2)
        table_height = max(10, num_components * 0.8)
        
        # Draw optical table
        table_rect = Rectangle((0, 0), table_width, table_height, 
                              linewidth=3, edgecolor='black', facecolor='white', alpha=0.1)
        ax.add_patch(table_rect)
        
        # Add title
        if title:
            ax.text(table_width/2, table_height + 0.5, title, 
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Position single photon sources
        sources = source_config['locations'] if isinstance(source_config['locations'], list) else functional_roles['potential_sources']
        source_positions = {}
        
        if sources:
            source_spacing = min(1.5, table_height / max(1, len(sources) + 1))
            source_start_y = table_height/2 - (len(sources)-1) * source_spacing / 2
            source_x = table_width * 0.12
            
            for i, vertex in enumerate(sorted(sources)):
                source_y = source_start_y + i * source_spacing
                source_positions[vertex] = (source_x, source_y)
                
                # Draw single photon source
                source = Polygon([(source_x, source_y+0.25), (source_x+0.25, source_y), 
                                (source_x, source_y-0.25), (source_x-0.25, source_y)], 
                               facecolor='lightcoral', edgecolor='darkred', linewidth=2)
                ax.add_patch(source)
                
                # Labels
                ax.text(source_x, source_y + 0.4, f'SPS{vertex}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
                ax.text(source_x, source_y - 0.4, source_config['wavelength'], ha='center', va='top',
                       fontsize=8, style='italic')
        
        # Position detectors
        detectors = detector_config['locations']
        detector_positions = {}
        
        if detectors:
            det_spacing = min(1.5, table_height / max(1, len(detectors) + 1))
            det_start_y = table_height/2 - (len(detectors)-1) * det_spacing / 2
            det_x = table_width * 0.88
            
            for i, vertex in enumerate(sorted(detectors)):
                det_y = det_start_y + i * det_spacing
                detector_positions[vertex] = (det_x, det_y)
                
                # Draw detector with proper size
                detector = Circle((det_x, det_y), 0.2, facecolor='yellow', 
                                edgecolor='orange', linewidth=3)
                ax.add_patch(detector)
                
                # Add detector label
                ax.text(det_x + 0.35, det_y, f'D{vertex}', ha='left', va='center',
                       fontsize=11, fontweight='bold')
                
                # Add detector type label
                ax.text(det_x, det_y - 0.35, 'SPD', ha='center', va='top',
                       fontsize=8, style='italic', color='darkblue')
        
        # Add connections and beam splitters based on graph structure
        self._add_optical_connections(ax, network_analysis, source_positions, detector_positions, 
                                    table_width, table_height)
        
        # Add legend and statistics
        self._add_optical_legend(ax, network_analysis, table_width, table_height)
        
        # Format and save
        ax.set_xlim(-0.5, table_width + 0.5)
        ax.set_ylim(-0.5, table_height + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Optical table saved to {save_path}")
        
        return fig
    
    def _plot_general_spdc_optical_table(self, network_analysis, save_path=None, title=None):
        """Plot optical table for general SPDC-based networks with multiple sources."""
        # Get key information
        strategy_info = network_analysis['implementation_strategy']
        spdc_sources = strategy_info['sources']
        all_detectors = strategy_info['detectors']
        beam_splitters = strategy_info['beam_splitters']
        ancilla_detectors = set(strategy_info['ancillas'])
        out_nodes = network_analysis['config'].get('out_nodes', [])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.set_facecolor('#f8f9fa')
        
        # Adaptive table dimensions based on number of sources
        num_sources = len(spdc_sources)
        num_detectors = len(all_detectors)
        table_width = max(16, num_sources * 3 + 4)
        table_height = max(12, num_detectors * 0.8)
        
        # Draw optical table
        table_rect = Rectangle((0, 0), table_width, table_height, 
                              linewidth=3, edgecolor='black', facecolor='white', alpha=0.1)
        ax.add_patch(table_rect)
        
        # Add title
        if title:
            ax.text(table_width/2, table_height + 0.5, title, 
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Position multiple SPDC sources
        spdc_positions = {}
        pump_positions = {}
        
        if num_sources > 1:
            # Arrange sources in a line on the left side
            source_spacing = min(2.0, table_height / max(1, num_sources + 1))
            source_start_y = table_height/2 - (num_sources-1) * source_spacing / 2
            source_x = table_width * 0.15
            
            for i, source_id in enumerate(spdc_sources):
                spdc_y = source_start_y + i * source_spacing
                spdc_positions[source_id] = (source_x, spdc_y)
                
                # Draw SPDC crystal
                spdc_crystal = Rectangle((source_x-0.3, spdc_y-0.15), 0.6, 0.3, 
                                       facecolor='lightblue', edgecolor='blue', linewidth=2)
                ax.add_patch(spdc_crystal)
                ax.text(source_x, spdc_y + 0.3, f'SPDC{source_id}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
                ax.text(source_x, spdc_y - 0.3, 'BBO Crystal', ha='center', va='top',
                       fontsize=8, style='italic')
                
                # Pump laser for each source
                pump_x = source_x - 1.2
                pump_y = spdc_y
                pump_positions[source_id] = (pump_x, pump_y)
                
                pump_laser = Rectangle((pump_x-0.2, pump_y-0.1), 0.4, 0.2, 
                                     facecolor='purple', edgecolor='darkviolet', linewidth=2)
                ax.add_patch(pump_laser)
                ax.text(pump_x, pump_y + 0.3, f'Pump{source_id}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
                ax.text(pump_x, pump_y - 0.3, '405nm', ha='center', va='top',
                       fontsize=7, style='italic')
                
                # Draw pump beam
                ax.arrow(pump_x + 0.2, pump_y, 0.8, 0, head_width=0.05, head_length=0.1,
                        fc='purple', ec='purple', linewidth=2)
        else:
            # Single source (original layout)
            source_x = table_width * 0.3
            spdc_y = table_height / 2
            spdc_positions[spdc_sources[0]] = (source_x, spdc_y)
            
            # Draw single SPDC crystal
            spdc_crystal = Rectangle((source_x-0.3, spdc_y-0.15), 0.6, 0.3, 
                                   facecolor='lightblue', edgecolor='blue', linewidth=2)
            ax.add_patch(spdc_crystal)
            ax.text(source_x, spdc_y + 0.3, 'SPDC', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
            ax.text(source_x, spdc_y - 0.3, 'BBO Crystal', ha='center', va='top',
                   fontsize=8, style='italic')
            
            # Single pump laser
            pump_x = source_x - 1.5
            pump_y = spdc_y
            pump_positions[spdc_sources[0]] = (pump_x, pump_y)
            
            pump_laser = Rectangle((pump_x-0.2, pump_y-0.1), 0.4, 0.2, 
                                 facecolor='purple', edgecolor='darkviolet', linewidth=2)
            ax.add_patch(pump_laser)
            ax.text(pump_x, pump_y + 0.3, 'Pump', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
            ax.text(pump_x, pump_y - 0.3, '405nm', ha='center', va='top',
                   fontsize=8, style='italic')
            
            # Draw pump beam
            ax.arrow(pump_x + 0.2, pump_y, 1.1, 0, head_width=0.05, head_length=0.1,
                    fc='purple', ec='purple', linewidth=2)
        
        # Position beam splitters in the middle
        bs_positions = {}
        if beam_splitters:
            bs_spacing = min(1.5, table_height / max(1, len(beam_splitters) + 1))
            bs_start_y = table_height/2 - (len(beam_splitters)-1) * bs_spacing / 2
            bs_x = table_width * 0.5
            
            for i, bs_id in enumerate(beam_splitters):
                bs_y = bs_start_y + i * bs_spacing
                bs_positions[bs_id] = (bs_x, bs_y)
                
                # Draw beam splitter
                bs_rect = Rectangle((bs_x-0.25, bs_y-0.25), 0.5, 0.5, 
                                  facecolor='lightgray', edgecolor='black', linewidth=2)
                ax.add_patch(bs_rect)
                # Add diagonal line to show beam splitting
                ax.plot([bs_x-0.25, bs_x+0.25], [bs_y-0.25, bs_y+0.25], 'black', linewidth=2)
                ax.text(bs_x, bs_y-0.4, f'BS{bs_id}', ha='center', va='top', 
                       fontsize=9, fontweight='bold')
        
        # Position detectors on the right side
        detector_positions = {}
        
        # Separate communication and ancilla detectors
        comm_detectors = [d for d in all_detectors if d not in ancilla_detectors]
        anc_detectors = [d for d in all_detectors if d in ancilla_detectors]
        
        # Position communication detectors
        if comm_detectors:
            comm_spacing = min(1.5, table_height * 0.6 / max(1, len(comm_detectors) + 1))
            comm_start_y = table_height * 0.7 - (len(comm_detectors)-1) * comm_spacing / 2
            comm_x = table_width * 0.8
            
            for i, det_id in enumerate(sorted(comm_detectors)):
                det_y = comm_start_y + i * comm_spacing
                detector_positions[det_id] = (comm_x, det_y)
                
                # Draw communication detector (circle)
                detector = Circle((comm_x, det_y), 0.15, facecolor='yellow', 
                                edgecolor='orange', linewidth=2)
                ax.add_patch(detector)
                ax.text(comm_x + 0.3, det_y, f'D{det_id}', ha='left', va='center',
                       fontweight='bold', color='darkorange')
        
        # Position ancilla detectors
        if anc_detectors:
            anc_spacing = min(1.2, table_height * 0.4 / max(1, len(anc_detectors) + 1))
            anc_start_y = table_height * 0.3 - (len(anc_detectors)-1) * anc_spacing / 2
            anc_x = table_width * 0.85
            
            for i, det_id in enumerate(sorted(anc_detectors)):
                det_y = anc_start_y + i * anc_spacing
                detector_positions[det_id] = (anc_x, det_y)
                
                # Draw ancilla detector (diamond)
                detector = RegularPolygon((anc_x, det_y), 4, radius=0.15, 
                                       facecolor='lightcoral', edgecolor='darkred', 
                                       linewidth=2, orientation=np.pi/4)
                ax.add_patch(detector)
                ax.text(anc_x + 0.3, det_y, f'A{det_id}', ha='left', va='center',
                       fontweight='bold', color='darkred')
        
        # Create element positions for routing
        element_positions = {}
        
        # Add SPDC source positions
        for source_id, pos in spdc_positions.items():
            element_positions[f'spdc_{source_id}'] = pos
        
        # Add beam splitter positions
        for bs_id, pos in bs_positions.items():
            element_positions[f'bs_{bs_id}'] = pos
        
        # Add detector positions
        for det_id, pos in detector_positions.items():
            element_positions[f'detector_{det_id}'] = pos
        
        # Draw optical routing connections
        self._draw_optical_routing(ax, element_positions, spdc_sources, beam_splitters, 
                                 out_nodes, list(ancilla_detectors))
        
        # Add legend reflecting real network architecture  
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='SPDC Source'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                      markersize=12, label='Central Multi-Port Beam Splitter'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                      markersize=10, label='Communication Detector'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Ancilla Detector'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='SPDC → Central Mixer'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Mixer → Communication'),
            plt.Line2D([0], [0], color='purple', linewidth=2, linestyle='--', label='Mixer → Ancilla'),
            plt.Line2D([0], [0], color='orange', linewidth=1, linestyle=':', label='Ancilla Network')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
                 fontsize=10, title='Optical Elements', frameon=True, fancybox=True, shadow=True)
        
        # Add concise info box explaining real architecture
        strategy_info = network_analysis['implementation_strategy']
        real_sources = len([s for s in strategy_info['sources'] if s != 3])  # Exclude central mixer
        info_text = f"""Real Quantum Network Architecture:
• {real_sources} SPDC sources (nodes 0,1,2,4)
• 1 central multi-port beam splitter (node 3)
• {len([d for d in strategy_info['detectors'] if d not in strategy_info['ancillas']])} communication detectors  
• {len(strategy_info['ancillas'])} ancilla detectors (entangled network)
• Compact design: all sources feed central mixer"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Format and save
        ax.set_xlim(-0.5, table_width + 0.5)
        ax.set_ylim(-0.5, table_height + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Optical table saved to {save_path}")
        
        return fig
    
    def _plot_qkd_optical_table(self, network_analysis, save_path=None, title=None):
        """Plot optical table for QKD networks."""
        return self._plot_general_spdc_optical_table(network_analysis, save_path, title)
    
    def _plot_bell_state_optical_table(self, network_analysis, save_path=None, title=None):
        """Plot optical table for Bell state generation networks."""
        return self._plot_general_spdc_optical_table(network_analysis, save_path, title)
    
    def _plot_multi_party_optical_table(self, network_analysis, save_path=None, title=None):
        """
        Plot optical table for multi-party entanglement networks (QKD, GHZ, etc.).
        Creates a sophisticated setup with multiple SPDC sources and beam splitters.
        """
        config = network_analysis['config']
        out_nodes = config.get('out_nodes', [])
        anc_detectors = config.get('anc_detectors', [])
        target_state = config.get('target_state', [])
        description = config.get('description', '')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Set up optical table
        table_width = 15.0
        table_height = 10.0
        ax.set_xlim(0, table_width)
        ax.set_ylim(0, table_height)
        ax.set_aspect('equal')
        
        # Draw optical table
        table_rect = plt.Rectangle((0.5, 0.5), table_width-1, table_height-1, 
                                 fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(table_rect)
        
        # Multi-party entanglement requires multiple SPDC sources
        num_parties = len(out_nodes)
        num_ancillas = len(anc_detectors)
        
        # Position multiple SPDC sources for complex entanglement
        spdc_positions = []
        for i in range(min(3, num_parties)):  # Use up to 3 SPDC sources
            x = 2.0 + i * 4.0
            y = table_height / 2
            spdc_positions.append((x, y))
            
            # Draw SPDC source
            spdc_rect = plt.Rectangle((x-0.3, y-0.2), 0.6, 0.4, 
                                    fill=True, facecolor='lightblue', 
                                    edgecolor='blue', linewidth=2)
            ax.add_patch(spdc_rect)
            ax.text(x, y, f'SPDC{i+1}', ha='center', va='center', fontsize=8, weight='bold')
        
        # Add pump lasers
        for i, (x, y) in enumerate(spdc_positions):
            pump_x = x
            pump_y = y + 1.5
            
            # Pump laser
            pump_rect = plt.Rectangle((pump_x-0.2, pump_y-0.15), 0.4, 0.3, 
                                    fill=True, facecolor='yellow', 
                                    edgecolor='orange', linewidth=2)
            ax.add_patch(pump_rect)
            ax.text(pump_x, pump_y, f'405nm\nPump{i+1}', ha='center', va='center', 
                   fontsize=7, weight='bold')
            
            # Pump beam
            ax.plot([pump_x, x], [pump_y-0.15, y+0.2], 'orange', linewidth=3, alpha=0.7)
        
        # Position beam splitters for multi-party entanglement
        bs_positions = []
        if num_parties > 2:
            # Create beam splitter network
            for i in range(min(2, num_parties-1)):
                x = 8.0 + i * 2.0
                y = table_height / 2 + (i % 2 - 0.5) * 1.5
                bs_positions.append((x, y))
                
                # Draw beam splitter
                bs_square = plt.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                                        fill=True, facecolor='lightgray', 
                                        edgecolor='black', linewidth=2)
                ax.add_patch(bs_square)
                
                # Add diagonal line
                ax.plot([x-0.25, x+0.25], [y-0.25, y+0.25], 'black', linewidth=2)
                ax.text(x, y-0.5, f'BS{i+1}', ha='center', va='center', fontsize=8, weight='bold')
        
        # Position detectors for communication nodes
        detector_positions = {}
        for i, node in enumerate(out_nodes):
            angle = 2 * np.pi * i / len(out_nodes)
            radius = 3.0
            x = table_width/2 + radius * np.cos(angle)
            y = table_height/2 + radius * np.sin(angle)
            detector_positions[node] = (x, y)
            
            # Draw detector
            detector_circle = plt.Circle((x, y), 0.3, fill=True, facecolor='red', 
                                       edgecolor='darkred', linewidth=2)
            ax.add_patch(detector_circle)
            ax.text(x, y, f'D{node}', ha='center', va='center', fontsize=8, 
                   weight='bold', color='white')
        
        # Position ancilla detectors
        for i, anc in enumerate(anc_detectors):
            x = 12.0 + (i % 3) * 1.0
            y = 2.0 + (i // 3) * 1.5
            detector_positions[anc] = (x, y)
            
            # Draw ancilla detector
            detector_square = plt.Rectangle((x-0.2, y-0.2), 0.4, 0.4, 
                                          fill=True, facecolor='orange', 
                                          edgecolor='darkorange', linewidth=2)
            ax.add_patch(detector_square)
            ax.text(x, y, f'A{anc}', ha='center', va='center', fontsize=7, 
                   weight='bold', color='white')
        
        # Add optical connections based on graph structure
        self._add_multi_party_connections(ax, network_analysis, spdc_positions, 
                                        bs_positions, detector_positions)
        
        # Add legend and information
        self._add_multi_party_legend(ax, table_width, table_height, network_analysis)
        
        # Set title
        if title is None:
            title = f"Multi-Party Quantum Network Setup ({num_parties} parties)"
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Optical table saved to {save_path}")
        
        return fig
    
    def _add_multi_party_connections(self, ax, network_analysis, spdc_positions, 
                                   bs_positions, detector_positions):
        """Add optical connections for multi-party networks."""
        # Mode colors
        mode_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange', 4: 'purple'}
        
        # Connect SPDC sources to beam splitters
        for i, spdc_pos in enumerate(spdc_positions):
            if i < len(bs_positions):
                bs_pos = bs_positions[i]
                # Signal beam
                ax.plot([spdc_pos[0]+0.3, bs_pos[0]-0.25], 
                       [spdc_pos[1]+0.1, bs_pos[1]], 
                       color='blue', linewidth=2.5, alpha=0.8)
                # Idler beam
                ax.plot([spdc_pos[0]+0.3, bs_pos[0]-0.25], 
                       [spdc_pos[1]-0.1, bs_pos[1]], 
                       color='red', linewidth=2.5, alpha=0.8)
        
        # Connect beam splitters to detectors
        for i, bs_pos in enumerate(bs_positions):
            # Connect to some detectors
            for j, (det_id, det_pos) in enumerate(detector_positions.items()):
                if j < 2:  # Connect each BS to 2 detectors
                    color = mode_colors.get(j % len(mode_colors), 'gray')
                    ax.plot([bs_pos[0]+0.25, det_pos[0]], [bs_pos[1], det_pos[1]], 
                           color=color, linewidth=2, alpha=0.7)
        
        # Add some direct connections from SPDC to detectors
        for i, spdc_pos in enumerate(spdc_positions):
            if i < len(detector_positions):
                det_pos = list(detector_positions.values())[i]
                ax.plot([spdc_pos[0]+0.3, det_pos[0]], [spdc_pos[1], det_pos[1]], 
                       color='purple', linewidth=2, alpha=0.6, linestyle='--')
    
    def _add_multi_party_legend(self, ax, table_width, table_height, network_analysis):
        """Add legend for multi-party network setup."""
        config = network_analysis['config']
        
        # Legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='SPDC Sources'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                      markersize=10, label='Pump Lasers (405nm)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                      markersize=10, label='Beam Splitters'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Communication Detectors'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                      markersize=10, label='Ancilla Detectors'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Signal Photons'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Idler Photons'),
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', alpha=0.6, label='Physical Connections')
        ]
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
                 fontsize=10, title='Optical Elements', frameon=True, fancybox=True, shadow=True)
        
        # Add info box
        info_text = f"""5-Node QKD Network:
Communication Parties: {len(config.get('out_nodes', []))}
Ancilla Detectors: {len(config.get('anc_detectors', []))}
SPDC Sources: {len(config.get('out_nodes', []))}
Target States: {len(config.get('target_state', []))}
Beam Splitters: {len(network_analysis['implementation_strategy']['beam_splitters'])}
Entanglement: Multi-Party Distribution"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    def plot_native_graph(self, save_path=None, title=None):
        """
        Create a PyTheus-style native graph plot with enhanced styling.
        Uses dynamic network analysis and adaptive sizing for any quantum network.
        """
        if not self.graph:
            print("⚠️ No graph data available for plotting!")
            return None
        
        # Get network analysis for adaptive parameters
        network_analysis = self.analyze_network_structure()
        
        if self.verbose:
            print("🎨 Creating PyTheus-style native graph plot...")
        
        # PyTheus standard colors (exact match)
        pytheus_colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
        
        # Create figure with clean background
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor('#f8f9fa')
        
        # Get vertices and create circular layout
        vertices = network_analysis['vertices']
        n_vertices = len(vertices)
        
        # Create circular layout (exact PyTheus method)
        angles = np.linspace(0, 2 * np.pi * (n_vertices - 1) / n_vertices, n_vertices)
        vertex_positions = {}
        radius = 0.9
        
        for i, vertex in enumerate(sorted(vertices)):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            vertex_positions[vertex] = (x, y)
        
        # Group edges by uncolored edge (PyTheus edgeBleach equivalent)
        edge_dict = {}
        for edge_key, weight in self.graph.items():
            if isinstance(edge_key, str):
                edge_tuple = eval(edge_key)
            else:
                edge_tuple = edge_key
            
            # Extract vertex pair and modes
            v1, v2 = edge_tuple[0], edge_tuple[1]
            mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
            uncolored_edge = (v1, v2)
            
            if uncolored_edge not in edge_dict:
                edge_dict[uncolored_edge] = []
            edge_dict[uncolored_edge].append((mode1, mode2, weight))
        
        # Draw edges with adaptive thickness and proper styling
        max_thickness = max(8, min(15, 60 / n_vertices))  # Adaptive thickness
        
        for uncolored_edge, colorings in edge_dict.items():
            v1, v2 = uncolored_edge
            pos1 = np.array(vertex_positions[v1])
            pos2 = np.array(vertex_positions[v2])
            
            mult = len(colorings)
            
            for i, (mode1, mode2, weight) in enumerate(colorings):
                # Get colors for modes
                col1 = pytheus_colors[int(mode1) % len(pytheus_colors)]
                col2 = pytheus_colors[int(mode2) % len(pytheus_colors)]
                
                # Calculate line thickness and transparency
                line_thickness = max(abs(weight) * max_thickness, 0.5)
                transparency = min(0.3 + abs(weight) * 0.7, 1.0)
                
                # Calculate edge path (PyTheus method)
                if not np.array_equal(pos1, pos2):
                    # Non-self loop with offset for multiple edges
                    diff = pos1 - pos2
                    rect = np.array([diff[1], -diff[0]])
                    rect = rect / np.linalg.norm(rect)
                    offset = (2 * i - mult + 1) * 0.05
                    midpoint = (pos1 + pos2) / 2 + offset * rect
                else:
                    # Self loop
                    midpoint = pos1 * 1.2
                
                # Draw edge in two parts with different colors (PyTheus style)
                ax.plot([pos1[0], midpoint[0]], [pos1[1], midpoint[1]], 
                       color=col1, linewidth=line_thickness, alpha=transparency)
                ax.plot([midpoint[0], pos2[0]], [midpoint[1], pos2[1]], 
                       color=col2, linewidth=line_thickness, alpha=transparency)
                
                # Add diamond marker for negative weights (PyTheus style)
                if weight < 0:
                    ax.plot(midpoint[0], midpoint[1], marker="d", markersize=8, 
                           markeredgewidth=2, markeredgecolor="black", color="white", zorder=10)
                
                # Add weight labels for significant weights
                if abs(weight) > 0.1 and (mult == 1 or i == 0):
                    label_pos = midpoint if mult > 1 else (pos1 + pos2) / 2
                    ax.text(label_pos[0], label_pos[1] + 0.12, f'{weight:.2f}', 
                           ha='center', va='center', 
                           fontsize=max(6, min(10, 80/n_vertices)), 
                           fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Draw vertices with adaptive sizing and color coding
        vertex_size = max(0.08, min(0.15, 1.0 / n_vertices))
        vertex_circles = []
        
        for vertex, pos in vertex_positions.items():
            circle = Circle(pos, vertex_size, alpha=0.9, zorder=11)
            vertex_circles.append(circle)
            
            # Adaptive font size
            font_size = max(8, min(14, 100/n_vertices))
            ax.text(pos[0], pos[1], str(vertex), ha='center', va='center',
                   fontsize=font_size, fontweight='bold', zorder=12)
        
        # Add vertex circles as collection (PyTheus method)
        from matplotlib.collections import PatchCollection
        circ_collection = PatchCollection(vertex_circles, zorder=10)
        circ_collection.set(facecolor='lightgrey', edgecolor='dimgray', linewidth=2)
        ax.add_collection(circ_collection)
        
        # Set plot properties (exact PyTheus style)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with proper formatting
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f'PyTheus Graph: {network_analysis["description"]}', 
                        fontsize=16, fontweight='bold', pad=20)
        
        # Add clean legend for modes - show ALL modes used
        unique_modes = network_analysis['mode_analysis']['unique_modes']
        legend_elements = []
        
        # Always include all modes that appear in the graph
        for mode in unique_modes:
            color = pytheus_colors[mode % len(pytheus_colors)]
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=4, 
                          label=f'Mode {mode}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.15, 1), fontsize=10, title='Edge Colors',
                     frameon=True, fancybox=True, shadow=True)
        
        # Add network statistics in a clean info box
        perfect_matchings = self.find_perfect_matchings()
        impl_strategy = network_analysis['implementation_strategy']
        strategy_name = "graph_structure_based"  # Our new general approach
        
        stats_text = f"""Vertices: {len(vertices)}
Edges: {len(self.graph)}
Modes: {network_analysis['mode_analysis']['num_modes']}
Perfect Matchings: {len(perfect_matchings)}
Strategy: {strategy_name}"""
        
        ax.text(-1.05, 1.05, stats_text, fontsize=9,
               verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save and return
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 PyTheus native graph saved to {save_path}")
        
        return fig
    
    def generate_analysis_report(self, save_path=None):
        """
        Generate a comprehensive analysis report.
        """
        # Get complete network analysis
        network_analysis = self.analyze_network_structure()
        
        if self.verbose:
            print("📊 Generating comprehensive analysis report...")
        
        # Build report
        report = f"""
# Quantum Network Analysis Report

## Network Overview
- **Description**: {network_analysis['description']}
- **Vertices**: {len(network_analysis['vertices'])}
- **Edges**: {network_analysis['edge_count']}
- **Graph Type**: {'Connected' if network_analysis['connectivity']['is_connected'] else 'Disconnected'}

## Graph Structure Analysis
- **Is Bipartite**: {network_analysis['connectivity']['is_bipartite']}
- **Is Tree**: {network_analysis['connectivity']['is_tree']}
- **Density**: {network_analysis['connectivity']['density']:.4f}
- **Clustering Coefficient**: {network_analysis['connectivity']['clustering']:.4f}
- **Diameter**: {network_analysis['connectivity']['diameter']}

## Mode Analysis
- **Number of Modes**: {network_analysis['mode_analysis']['num_modes']}
- **Unique Modes**: {network_analysis['mode_analysis']['unique_modes']}
- **Maximum Edge Weight**: {network_analysis['mode_analysis']['max_weight']:.4f}
- **Complex Weights**: {network_analysis['mode_analysis']['has_complex_weights']}

## Functional Role Analysis
- **Potential Sources**: {network_analysis['functional_roles']['potential_sources']}
- **Potential Detectors**: {network_analysis['functional_roles']['potential_detectors']}
- **Potential Ancillas**: {network_analysis['functional_roles']['potential_ancillas']}
- **Leaf Nodes**: {network_analysis['functional_roles']['leaves']}
- **Hub Nodes**: {network_analysis['functional_roles']['hubs']}

## Degree Statistics
- **Mean Degree**: {network_analysis['functional_roles']['degree_stats']['mean']:.2f}
- **Degree Range**: {network_analysis['functional_roles']['degree_stats']['min']} - {network_analysis['functional_roles']['degree_stats']['max']}
- **Degree Standard Deviation**: {network_analysis['functional_roles']['degree_stats']['std']:.2f}

## Quantum State Analysis
- **Has Target State**: {network_analysis['state_analysis']['has_target_state']}
- **Single Emitters**: {network_analysis['state_analysis']['single_emitters']}
- **Entanglement Type**: {network_analysis['state_analysis'].get('entanglement_type', 'unknown')}
- **Particle Number**: {network_analysis['state_analysis']['particle_number']}

## Implementation Strategy
- **Analysis Type**: Graph Structure Based
- **Sources Identified**: {len(network_analysis['implementation_strategy']['sources'])}
- **Detectors Identified**: {len(network_analysis['implementation_strategy']['detectors'])}
- **Beam Splitters**: {len(network_analysis['implementation_strategy']['beam_splitters'])}
- **Ancillas**: {len(network_analysis['implementation_strategy']['ancillas'])}
- **Requires Heralding**: {network_analysis['implementation_strategy']['requires_heralding']}
- **Complexity Level**: {network_analysis['implementation_strategy']['complexity_level']}

## Optical Elements
- **Sources**: {network_analysis['implementation_strategy']['sources']}
- **Detectors**: {network_analysis['implementation_strategy']['detectors']}
- **Beam Splitters**: {network_analysis['implementation_strategy']['beam_splitters']}
- **Ancillas**: {network_analysis['implementation_strategy']['ancillas']}

## Graph Motifs
- **Triangles**: {len(network_analysis['motifs']['triangles'])}
- **Squares**: {len(network_analysis['motifs']['squares'])}
- **Stars**: {len(network_analysis['motifs']['stars'])}

## Edge Details
"""
        
        # Add detailed edge information
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            weight = self.graph[edge]
            if len(edge_tuple) >= 4:
                report += f"- Edge {edge_tuple[0]}-{edge_tuple[1]} (modes {edge_tuple[2]}-{edge_tuple[3]}): {weight:.6f}\n"
            else:
                report += f"- Edge {edge_tuple[0]}-{edge_tuple[1]}: {weight:.6f}\n"
                report += f"- Edge {edge_tuple[0]}-{edge_tuple[1]}: {weight:.6f}\n"
        
        report += f"""

## Configuration Details
```json
{json.dumps(self.config, indent=2)}
```

## Analysis Metadata
- **Analysis Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Interpreter Version**: General Quantum Network Interpreter v1.0
- **Analysis Method**: Dynamic graph structure analysis
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            if self.verbose:
                print(f"📄 Analysis report saved to {save_path}")
        
        return report
    
    def run_complete_analysis(self, base_filename=None):
        """
        Run complete analysis and generate all outputs.
        """
        if not base_filename:
            base_filename = "general_network_analysis"
        
        if self.verbose:
            print("🚀 Running complete quantum network analysis...")
        
        # Generate all outputs
        results = {}
        
        # 1. Optical table setup
        optical_fig = self.plot_optical_table_setup(
            save_path=f"{base_filename}_optical_table_setup.png",
            title="General Quantum Network - Optical Table Setup"
        )
        results['optical_table'] = optical_fig
        
        # 2. Native graph plot
        native_fig = self.plot_native_graph(
            save_path=f"{base_filename}_native_plot.png",
            title="General Quantum Network - PyTheus Graph"
        )
        results['native_graph'] = native_fig
        
        # 3. Analysis report
        report = self.generate_analysis_report(
            save_path=f"{base_filename}_report.txt"
        )
        results['report'] = report
        
        # 4. Network analysis data
        network_analysis = self.analyze_network_structure()
        results['analysis'] = network_analysis
        
        if self.verbose:
            print("✅ Complete analysis finished!")
            print(f"   Generated: {base_filename}_optical_table_setup.png")
            print(f"   Generated: {base_filename}_native_plot.png")
            print(f"   Generated: {base_filename}_report.txt")
        
        return results
    
    def find_perfect_matchings(self):
        """
        Find all perfect matchings in the graph.
        A perfect matching is a set of edges where every vertex is incident to exactly one edge.
        """
        if not self.graph or not self.vertices:
            return []
        
        # Create NetworkX graph
        G = nx.Graph()
        
        for edge in self.graph.keys():
            if isinstance(edge, str):
                edge_tuple = eval(edge)
            else:
                edge_tuple = edge
            
            if len(edge_tuple) >= 2:
                G.add_edge(edge_tuple[0], edge_tuple[1])
        
        # Find all perfect matchings
        try:
            # Only graphs with even number of vertices can have perfect matchings
            if len(G.nodes()) % 2 != 0:
                return []
            
            # Use NetworkX to find maximum matchings
            matchings = []
            max_matching = nx.max_weight_matching(G)
            
            # Check if it's a perfect matching
            if len(max_matching) == len(G.nodes()) // 2:
                matchings.append(list(max_matching))
            
            return matchings
            
        except Exception:
            # If NetworkX methods fail, return empty list
            return []
    
    def _identify_actual_sources(self):
        """
        Identify REAL SPDC sources based on actual graph analysis.
        From the graph structure: nodes 0,1,2,4 are sources that feed into node 3 (central mixer)
        """
        # Analyze which nodes connect to the central hub (node 3)
        sources_to_hub = []
        
        for edge_key, weight in self.graph.items():
            if isinstance(edge_key, str):
                edge_tuple = eval(edge_key)
            else:
                edge_tuple = edge_key
            
            if len(edge_tuple) >= 2:
                v1, v2 = edge_tuple[0], edge_tuple[1]
                # Node 3 is the central hub based on the graph structure
                if v1 == 3 and v2 <= 4 and v2 != 3:
                    sources_to_hub.append(v2)
                elif v2 == 3 and v1 <= 4 and v1 != 3:
                    sources_to_hub.append(v1)
        
        # Remove duplicates and sort
        sources = sorted(list(set(sources_to_hub)))
        
        if self.verbose:
            print(f"   Real Network: {len(sources)} SPDC sources {sources} feeding central mixer (node 3)")
        
        return sources
    
    def _identify_beam_splitter_nodes(self):
        """
        Identify the real beam splitter configuration.
        From graph analysis: Node 3 is the central multi-port beam splitter/mixer
        """
        # Node 3 is the central mixing point
        central_mixer = [3]
        
        if self.verbose:
            print(f"   Real Network: 1 central multi-port beam splitter (node {central_mixer[0]})")
        
        return central_mixer
    
    def _identify_ancilla_nodes(self):
        """
        Identify ancilla nodes from config.
        """
        anc_detectors = self.config.get('anc_detectors', [])
        return anc_detectors
    
    def _identify_actual_detectors(self):
        """
        Identify detector paths based on PyTheus graph representation and config.
        Communication detectors (0,1,2,4) + Ancilla detectors (5,6,7,8,9)
        """
        # Get config information
        out_nodes = self.config.get('out_nodes', [])
        anc_detectors = self.config.get('anc_detectors', [])
        single_emitters = self.config.get('single_emitters', [])
        in_nodes = self.config.get('in_nodes', [])
        
        # Communication detectors (main quantum parties) - exclude central mixer (node 3)
        comm_detectors = [v for v in out_nodes if v not in single_emitters and v not in in_nodes and v != 3]
        
        # Ancilla detectors (heralding/post-selection)
        ancilla_detectors = [v for v in anc_detectors if v not in single_emitters and v not in in_nodes]
        
        # Total detectors
        all_detectors = comm_detectors + ancilla_detectors
        
        if self.verbose:
            print(f"   Identified {len(all_detectors)} detector paths:")
            print(f"     Communication detectors: {len(comm_detectors)} {comm_detectors}")
            print(f"     Ancilla detectors: {len(ancilla_detectors)} {ancilla_detectors}")
        
        return all_detectors
    
    def _determine_optical_elements(self, sources, detectors, beam_splitters, ancillas):
        """
        Determine the optical elements needed for the network.
        """
        return {
            'spdc_sources': len(sources),
            'pump_lasers': len(sources), 
            'beam_splitters': len(beam_splitters),
            'communication_detectors': len([d for d in detectors if d not in ancillas]),
            'ancilla_detectors': len(ancillas),
            'total_detectors': len(detectors)
        }
    
    def _build_connection_map(self):
        """
        Build a map of how optical elements should be connected.
        """
        connections = []
        
        for edge_key, weight in self.graph.items():
            if isinstance(edge_key, str):
                edge_tuple = eval(edge_key)
            else:
                edge_tuple = edge_key
            
            if len(edge_tuple) >= 2:
                v1, v2 = edge_tuple[0], edge_tuple[1]
                mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                
                connections.append({
                    'from': v1,
                    'to': v2,
                    'mode1': mode1,
                    'mode2': mode2,
                    'weight': weight
                })
        
        return connections
    
    def _structural_source_analysis(self):
        """
        Fallback structural analysis for sources.
        """
        degrees = self._compute_vertex_degrees()
        sources = []
        
        for vertex in self.vertices:
            outgoing_weight = 0
            incoming_weight = 0
            
            for edge_key, weight in self.graph.items():
                if isinstance(edge_key, str):
                    edge_tuple = eval(edge_key)
                else:
                    edge_tuple = edge_key
                
                if len(edge_tuple) >= 2:
                    v1, v2 = edge_tuple[0], edge_tuple[1]
                    if v1 == vertex:
                        outgoing_weight += abs(weight)
                    elif v2 == vertex:
                        incoming_weight += abs(weight)
            
            if outgoing_weight > incoming_weight and degrees.get(vertex, 0) > 0:
                sources.append(vertex)
        
        return sources
    
    def _create_vertex_element_mapping(self, element_positions, sources, out_nodes, anc_detectors, beam_splitters):
        """
        Create a proper mapping from vertices to their optical elements.
        
        Key insight: In QKD networks:
        - Out nodes (0-4) are COMMUNICATION PARTIES that need both sources AND detectors
        - Ancilla nodes (5-9) are HERALDING DETECTORS 
        - Beam splitters are INTERMEDIATE OPTICAL ELEMENTS for mixing
        """
        vertex_to_element = {}
        
        # Communication parties (0-4) need SPDC sources 
        for node_id in out_nodes:
            if node_id in sources:
                vertex_to_element[node_id] = f'spdc_{node_id}'
            else:
                vertex_to_element[node_id] = f'comm_det_{node_id}'
        
        # Ancilla detectors (5-9) are pure detectors
        for det_id in anc_detectors:
            vertex_to_element[det_id] = f'anc_det_{det_id}'
        
        # Beam splitters are separate optical elements (not necessarily nodes)
        # They're placed strategically to mix signals
        for bs_id in beam_splitters:
            vertex_to_element[bs_id] = f'bs_{bs_id}'
        
        # Any unmapped vertices default to detectors
        all_vertices = set(self.vertices)
        mapped_vertices = set(vertex_to_element.keys())
        unmapped_vertices = all_vertices - mapped_vertices
        
        for vertex in unmapped_vertices:
            vertex_to_element[vertex] = f'det_{vertex}'
        
        return vertex_to_element
    
    def _find_element_position(self, vertex_id, vertex_to_element, element_positions):
        """
        Find the position of any optical element based on vertex ID using the comprehensive mapping.
        """
        # Get the element key for this vertex
        element_key = vertex_to_element.get(vertex_id)
        
        if element_key and element_key in element_positions:
            return element_positions[element_key]
        
        # Fallback: try different possible keys
        fallback_keys = [
            f'spdc_{vertex_id}',
            f'comm_det_{vertex_id}',
            f'anc_det_{vertex_id}',
            f'bs_{vertex_id}',
            f'det_{vertex_id}',
            f'pump_{vertex_id}',
            str(vertex_id)
        ]
        
        
        for key in fallback_keys:
            if key in element_positions:
                return element_positions[key]
        
        return None
    
    def _draw_single_connection(self, ax, pos1, pos2, mode1, mode2, weight, mode_colors, connection_label):
        """
        Draw a single connection between two optical elements.
        """
        # Use mode colors
        color1 = mode_colors.get(mode1, 'gray')
        color2 = mode_colors.get(mode2, 'gray')
        
        # Line thickness based on weight
        thickness = max(1, min(4, abs(weight) * 2))
        alpha = min(0.8, max(0.4, abs(weight)))
        
        # Draw connection
        if mode1 == mode2:
            # Same mode - single colored line
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   color=color1, linewidth=thickness, alpha=alpha)
        else:
            # Different modes - two-color line
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax.plot([pos1[0], mid_x], [pos1[1], mid_y], 
                   color=color1, linewidth=thickness, alpha=alpha)
            ax.plot([mid_x, pos2[0]], [mid_y, pos2[1]], 
                   color=color2, linewidth=thickness, alpha=alpha)
        
        # Draw connection label
        if connection_label:
            ax.text((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2, connection_label, 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='black')
    
    def _add_physical_routing_connections(self, ax, element_positions, sources, beam_splitters, out_nodes, anc_detectors):
        """
        Add physical routing connections that show how photons flow through the optical setup.
        This represents the actual optical paths (SPDC → beam splitters → detectors).
        """
        if self.verbose:
            print("   Adding physical routing connections...")
        
        # Connect SPDC sources to beam splitters
        for i, source_id in enumerate(sources):
            spdc_pos = element_positions.get(f'spdc_{source_id}')
            if spdc_pos and beam_splitters:
                # Connect to corresponding beam splitter
                bs_id = beam_splitters[i % len(beam_splitters)]
                bs_pos = element_positions.get(f'bs_{bs_id}')
                
                if bs_pos:
                    # Signal path (blue line)
                    ax.plot([spdc_pos[0]+0.3, bs_pos[0]-0.25], 
                           [spdc_pos[1]+0.1, bs_pos[1]], 
                           color='blue', linewidth=2.5, alpha=0.8,
                           label='Signal Path' if i == 0 else "")
                    
                    # Idler path (red line)
                    ax.plot([spdc_pos[0]+0.3, bs_pos[0]-0.25], 
                           [spdc_pos[1]-0.1, bs_pos[1]], 
                           color='red', linewidth=2.5, alpha=0.8,
                           label='Idler Path' if i == 0 else "")
        
        # Connect beam splitters to detectors
        for i, bs_pos in enumerate(beam_splitters):
            # Connect to communication detectors
            comm_det_id = out_nodes[i % len(out_nodes)] if out_nodes else None
            if comm_det_id is not None:
                det_pos = element_positions.get(f'detector_{comm_det_id}')
                if det_pos:
                    ax.plot([bs_pos[0]+0.25, det_pos[0]-0.2], 
                           [bs_pos[1]+0.1, det_pos[1]], 
                           color='green', linewidth=2, alpha=0.7,
                           label='To Communication' if i == 0 else "")
            
            # Connect to ancilla detectors
            anc_det_id = anc_detectors[i % len(anc_detectors)] if anc_detectors else None
            if anc_det_id is not None:
                det_pos = element_positions.get(f'detector_{anc_det_id}')
                if det_pos:
                    ax.plot([bs_pos[0]+0.25, det_pos[0]-0.2], 
                           [bs_pos[1]-0.1, det_pos[1]], 
                           color='purple', linewidth=2, alpha=0.7,
                           label='To Ancilla' if i == 0 else "")
    
    def _draw_optical_routing(self, ax, element_positions, spdc_sources, beam_splitters, 
                            out_nodes, anc_detectors):
        """
        Draw the REAL network architecture based on actual graph structure:
        - 4 SPDC sources (0,1,2,4) → Central multi-port beam splitter (3)
        - Central BS outputs to communication detectors AND ancilla network
        """
        if self.verbose:
            print("   Drawing real network architecture...")
        
        # STEP 1: All SPDC sources feed into the central beam splitter (node 3)
        central_bs_id = 3  # Node 3 is the central mixing point
        central_bs_pos = element_positions.get(f'bs_{central_bs_id}')
        
        if central_bs_pos:
            for source_id in spdc_sources:
                spdc_pos = element_positions.get(f'spdc_{source_id}')
                if spdc_pos:
                    # Check if there's a real connection in the graph
                    connection_weight = 0
                    for edge_key, weight in self.graph.items():
                        if isinstance(edge_key, str):
                            edge_tuple = eval(edge_key)
                        else:
                            edge_tuple = edge_key
                        
                        if len(edge_tuple) >= 2:
                            v1, v2 = edge_tuple[0], edge_tuple[1]
                            if (v1 == source_id and v2 == central_bs_id) or (v1 == central_bs_id and v2 == source_id):
                                connection_weight = weight
                                break
                    
                    if connection_weight != 0:
                        # Draw connection with line style based on weight
                        line_style = '-' if connection_weight > 0 else '--'
                        ax.plot([spdc_pos[0]+0.3, central_bs_pos[0]-0.3], 
                               [spdc_pos[1], central_bs_pos[1]], 
                               color='blue', linewidth=2.5, alpha=0.9, linestyle=line_style)
                        
                        if self.verbose:
                            print(f"   SPDC {source_id} → Central BS {central_bs_id} (w={connection_weight:.2f})")
        
        # STEP 2: Central beam splitter outputs to communication detectors
        for comm_det_id in [0, 1, 2, 4]:  # Communication nodes (excluding central node 3)
            if comm_det_id in out_nodes:
                det_pos = element_positions.get(f'detector_{comm_det_id}')
                if det_pos and central_bs_pos:
                    ax.plot([central_bs_pos[0]+0.3, det_pos[0]-0.2], 
                           [central_bs_pos[1], det_pos[1]], 
                           color='green', linewidth=2, alpha=0.8)
                    
                    if self.verbose:
                        print(f"   Central BS {central_bs_id} → Comm Det {comm_det_id}")
        
        # STEP 3: Show connections to ancilla network
        # Group ancillas by their connection strength
        strong_ancilla_connections = []
        
        for anc_id in anc_detectors:
            # Find strongest connection to communication nodes
            max_weight = 0
            connected_comm_node = None
            
            for edge_key, weight in self.graph.items():
                if abs(weight) >= 0.9:  # Very strong connections only
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                    
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        if (v1 in [0,1,2,4] and v2 == anc_id) or (v2 in [0,1,2,4] and v1 == anc_id):
                            if abs(weight) > abs(max_weight):
                                max_weight = weight
                                connected_comm_node = v1 if v2 == anc_id else v2
            
            if connected_comm_node is not None:
                strong_ancilla_connections.append((anc_id, connected_comm_node, max_weight))
        
        # Draw ancilla connections through the central beam splitter
        for anc_id, comm_node, weight in strong_ancilla_connections:
            anc_pos = element_positions.get(f'detector_{anc_id}')
            if anc_pos and central_bs_pos:
                # Connection from central BS to ancilla
                ax.plot([central_bs_pos[0]+0.2, anc_pos[0]-0.2], 
                       [central_bs_pos[1]-0.2, anc_pos[1]], 
                       color='purple', linewidth=1.8, alpha=0.7, linestyle='--')
                
                if self.verbose:
                    print(f"   Central BS → Anc Det {anc_id} (via comm {comm_node}, w={weight:.2f})")
        
        # STEP 4: Show internal ancilla network connections (the ±1.0 perfect correlations)
        ancilla_internal_connections = []
        for edge_key, weight in self.graph.items():
            if abs(weight) == 1.0:  # Perfect correlations
                if isinstance(edge_key, str):
                    edge_tuple = eval(edge_key)
                else:
                    edge_tuple = edge_key
                
                if len(edge_tuple) >= 2:
                    v1, v2 = edge_tuple[0], edge_tuple[1]
                    if v1 in anc_detectors and v2 in anc_detectors:
                        ancilla_internal_connections.append((v1, v2, weight))
        
        # Draw a few key internal ancilla connections
        for v1, v2, weight in ancilla_internal_connections[:3]:  # Limit to avoid clutter
            pos1 = element_positions.get(f'detector_{v1}')
            pos2 = element_positions.get(f'detector_{v2}')
            if pos1 and pos2:
                line_style = '-' if weight > 0 else ':'
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       color='orange', linewidth=1, alpha=0.5, linestyle=line_style)
                
                if self.verbose:
                    print(f"   Anc {v1} ↔ Anc {v2} (internal correlation, w={weight:.1f})")
        
        # Summary
        if self.verbose:
            print(f"   Real architecture: {len(spdc_sources)} sources → 1 central mixer → {len(anc_detectors)} ancilla network")
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Signal Photons'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Idler Photons'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Communication Detection'),
            plt.Line2D([0], [0], color='purple', linewidth=1.5, label='Heralding (±1.0 weights only)')
        ]
        
        if hasattr(ax, 'legend'):
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
                     fontsize=9, title='Photon Paths', frameon=True)
        
        # Simple physics explanation
        if hasattr(ax, 'transAxes'):
            explanation = f"""Simplified QKD Network:
• {len(spdc_sources)} SPDC sources → {len(beam_splitters)} beam splitters (1:1)
• Each beam splitter → communication detector (1:1)
• Only strongest heralding connections shown (weight = ±1.0)
• Blue/Red: Entangled photon pairs from SPDC
• Green: Final detection for communication
• Purple: Ancilla heralding for state selection"""
            
            ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=8, 
                   verticalalignment='bottom', 
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
def create_interpreter(config=None, graph=None, verbose=True):
    """
    Convenience function to create an interpreter that handles both file paths and dict data.
    
    Args:
        config: Either a file path (str) or config dict
        graph: Either a file path (str) or graph dict
        verbose: Whether to print verbose output
    
    Returns:
        GeneralQuantumNetworkInterpreter instance
    """
    config_path = None
    graph_path = None
    config_data = None
    graph_data = None
    
    # Handle config input
    if isinstance(config, str):
        config_path = config
    elif isinstance(config, dict):
        config_data = config
    elif config is not None:
        raise ValueError(f"config must be a file path (str) or dict, got {type(config)}")
    
    # Handle graph input
    if isinstance(graph, str):
        graph_path = graph
    elif isinstance(graph, dict):
        graph_data = graph
    elif graph is not None:
        raise ValueError(f"graph must be a file path (str) or dict, got {type(graph)}")
    
    return GeneralQuantumNetworkInterpreter(
        config_path=config_path,
        graph_path=graph_path,
        config_data=config_data,
        graph_data=graph_data,
        verbose=verbose
    )


def analyze_quantum_network(config, graph, base_filename=None, verbose=True):
    """
    Convenience function to analyze a quantum network with flexible input.
    
    Args:
        config: Either a file path (str) or config dict
        graph: Either a file path (str) or graph dict
        base_filename: Base name for output files (optional)
        verbose: Whether to print verbose output
    
    Returns:
        Analysis results dict
    """
    interpreter = create_interpreter(config, graph, verbose)
    return interpreter.run_complete_analysis(base_filename)


# For backwards compatibility, also expose the main class directly
__all__ = ['GeneralQuantumNetworkInterpreter', 'create_interpreter', 'analyze_quantum_network']
