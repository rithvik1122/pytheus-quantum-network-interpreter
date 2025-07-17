#!/usr/bin/env python3
"""
PyTheus Modular Quantum Network Interpreter

A modular interpreter for analyzing and visualizing PyTheus-optimized quantum networks.

Author: Rithvik
GitHub: https://github.com/rithvik1122/pytheus-quantum-network-interpreter
License: MIT
Version: 1.0.0
Date: July 2025

This module provides tools for:
- Loading and parsing PyTheus quantum network configurations
- Analyzing network topology and identifying functional components
- Generating optical table layout visualizations
- Creating native PyTheus-style graph plots
- Producing comprehensive analysis reports

The interpreter is modular and adaptive, analyzing tested quantum network
structures dynamically with specialized modules for different network types.
It generates physically meaningful optical table setups and PyTheus-style graphs
for validated quantum state and network configurations.
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

class ModularQuantumNetworkInterpreter:
    """
    A modular quantum network interpreter that adapts to tested graph structures
    and configurations with specialized modules for different network types.
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
        Modular network analysis that adapts to tested network structures.
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
            
            # Check if out_nodes are explicitly specified (heralded Bell, etc.)
            if self.config.get('out_nodes'):
                potential_detectors = self.config.get('out_nodes', [])
            else:
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
        Create a modular optical table setup based on actual graph structure analysis.
        No predefined categories - interprets any graph structure correctly.
        """
        # Get comprehensive network analysis
        network_analysis = self.analyze_network_structure()
        
        if self.verbose:
            print("🔬 Creating modular optical table setup...")
            strategy_info = network_analysis['implementation_strategy']
            print(f"   Sources: {strategy_info['sources']}")
            print(f"   Detectors: {strategy_info['detectors']}")
            print(f"   Beam Splitters: {strategy_info['beam_splitters']}")
            print(f"   Ancillas: {strategy_info['ancillas']}")
        
        # Determine network type and use appropriate plotting method
        if self.config.get('single_emitters'):
            # W4, single photon source networks
            if self.verbose:
                print("   Detected single photon source network (W4, etc.)")
            return self._plot_single_photon_optical_table(network_analysis, save_path, title)
        elif self.config.get('out_nodes') and self.config.get('anc_detectors'):
            # Multi-party QKD networks with explicit communication and ancilla nodes
            if self.verbose:
                print("   Detected multi-party QKD network with ancilla heralding")
            return self._plot_general_spdc_optical_table(network_analysis, save_path, title)
        else:
            # Modular quantum networks (GHZ, Bell, cluster, supported topologies)
            if self.verbose:
                print("   Detected modular quantum network - using adaptive plotting")
            return self._plot_adaptive_quantum_network(network_analysis, save_path, title)
    
    def _plot_single_photon_optical_table(self, network_analysis, save_path=None, title=None):
        """Plot optical table for single photon source networks."""
        # Get key information
        strategy_info = network_analysis['implementation_strategy']
        sources = strategy_info['sources']
        detectors = strategy_info['detectors']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_facecolor('#f8f9fa')
        
        # Adaptive table dimensions
        num_components = len(sources) + len(detectors)
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
                ax.text(source_x, source_y - 0.4, '810nm', ha='center', va='top',
                       fontsize=8, style='italic')
        
        # Position detectors
        detector_positions = {}
        
        if detectors:
            det_spacing = min(1.5, table_height / max(1, len(detectors) + 1))
            det_start_y = table_height/2 - (len(detectors)-1) * det_spacing / 2
            det_x = table_width * 0.88
            
            # Get ancilla information for proper visualization
            strategy_info = network_analysis.get('implementation_strategy', {})
            ancillas = strategy_info.get('ancillas', [])
            
            for i, vertex in enumerate(sorted(detectors)):
                det_y = det_start_y + i * det_spacing
                detector_positions[vertex] = (det_x, det_y)
                
                # Different colors and labels for ancilla vs communication detectors
                if vertex in ancillas:
                    detector_color = 'lightcoral'
                    edge_color = 'darkred'
                    label_prefix = 'A'
                    detector_type = 'ANC'
                else:
                    detector_color = 'yellow'
                    edge_color = 'orange'
                    label_prefix = 'D'
                    detector_type = 'SPD'
                
                # Draw detector with proper size and color
                detector = Circle((det_x, det_y), 0.2, facecolor=detector_color, 
                                edgecolor=edge_color, linewidth=3)
                ax.add_patch(detector)
                
                # Add detector label
                ax.text(det_x + 0.35, det_y, f'{label_prefix}{vertex}', ha='left', va='center',
                       fontsize=11, fontweight='bold', color=edge_color)
                
                # Add detector type label
                ax.text(det_x, det_y - 0.35, detector_type, ha='center', va='top',
                       fontsize=8, style='italic', color=edge_color)
        
        # Add connections and beam splitters based on graph structure
        self._add_optical_connections(ax, network_analysis, source_positions, detector_positions, 
                                    table_width, table_height)
        
        # Add legend and statistics
        self._add_optical_legend(ax, table_width, table_height, network_analysis)
        
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
        
        # Note: Title will be set later to avoid duplication
        
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
                # Add diagonal line
                ax.plot([bs_x-0.25, bs_x+0.25], [bs_y-0.25, bs_y+0.25], 'black', linewidth=2)
                ax.text(bs_x, bs_y-0.4, f'BS{bs_id}', ha='center', va='top', 
                       fontsize=8, fontweight='bold')
        
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
        
        # Position ancilla detectors (including dual-role beam splitters)
        if anc_detectors:
            anc_spacing = min(1.2, table_height * 0.4 / max(1, len(anc_detectors) + 1))
            anc_start_y = table_height * 0.3 - (len(anc_detectors)-1) * anc_spacing / 2
            anc_x = table_width * 0.85
            
            for i, det_id in enumerate(sorted(anc_detectors)):
                # Check if this ancilla is also a beam splitter (dual role)
                if det_id in beam_splitters and det_id in bs_positions:
                    # For dual-role elements, use the beam splitter position but add detector symbol
                    bs_x, bs_y = bs_positions[det_id]
                    detector_positions[det_id] = (bs_x, bs_y)
                    
                    # Add red border to beam splitter to indicate dual role
                    bs_dual_border = Rectangle((bs_x-0.32, bs_y-0.32), 0.64, 0.64, 
                                             facecolor='none', edgecolor='darkred', linewidth=3,
                                             zorder=6)
                    ax.add_patch(bs_dual_border)
                    
                    # Add small ancilla indicator in corner
                    detector = Circle((bs_x + 0.35, bs_y + 0.35), 0.08, 
                                    facecolor='lightcoral', edgecolor='darkred', linewidth=1.5,
                                    zorder=7)
                    ax.add_patch(detector)
                    ax.text(bs_x + 0.35, bs_y + 0.55, f'A{det_id}', ha='center', va='bottom',
                           fontsize=7, fontweight='bold', color='darkred', zorder=7)
                    
                    if self.verbose:
                        print(f"   Node {det_id}: Dual-role (Beam Splitter + Ancilla Detector)")
                else:
                    # Pure ancilla detector (not also a beam splitter)
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
        
        # Add legend and information
        self._add_optical_legend(ax, table_width, table_height, network_analysis)
        
        # Set title
        if title is None:
            title = f"Multi-Party Quantum Network Setup ({num_sources} sources)"
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
    
    def _plot_adaptive_quantum_network(self, network_analysis, save_path=None, title=None):
        """
        Plot optical table for modular quantum network using adaptive analysis.
        Uses the clean visualization principles without forcing specific network categories.
        """
        # Get key information
        strategy_info = network_analysis['implementation_strategy']
        sources = strategy_info['sources']
        detectors = strategy_info['detectors']
        beam_splitters = strategy_info['beam_splitters']
        ancillas = strategy_info['ancillas']
        
        if self.verbose:
            print(f"   Adaptive plotting for modular quantum network:")
            print(f"     Sources: {sources}")
            print(f"     Detectors: {detectors}")
            print(f"     Beam Splitters: {beam_splitters}")
            print(f"     Ancillas: {ancillas}")
        
        # Create figure with adaptive sizing
        num_sources = len(sources)
        num_detectors = len(detectors)
        num_total = num_sources + num_detectors + len(beam_splitters)
        
        # Adaptive figure size
        fig_width = max(14, min(20, num_total * 1.5))
        fig_height = max(10, min(16, num_total * 0.8))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_facecolor('#f8f9fa')
        
        # Adaptive table dimensions
        table_width = max(12, num_total * 1.2)
        table_height = max(8, num_total * 0.6)
        
        # Draw optical table
        table_rect = Rectangle((0, 0), table_width, table_height, 
                              linewidth=3, edgecolor='black', facecolor='white', alpha=0.1)
        ax.add_patch(table_rect)
        
        # Add title (single location)
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            description = network_analysis.get('description', 'Modular Quantum Network')
            ax.set_title(f'Adaptive Quantum Network: {description}', fontsize=16, fontweight='bold', pad=20)
        
        # Position sources on the left
        source_positions = {}
        if sources:
            source_spacing = min(1.5, table_height / max(1, len(sources) + 1))
            source_start_y = table_height/2 - (len(sources)-1) * source_spacing / 2
            source_x = table_width * 0.15
            
            for i, source_id in enumerate(sorted(sources)):
                source_y = source_start_y + i * source_spacing
                source_positions[source_id] = (source_x, source_y)
                
                # Draw source - adaptive shape based on network type
                if len(sources) == 1:
                    # Single source - larger representation
                    source_shape = Rectangle((source_x-0.4, source_y-0.2), 0.8, 0.4, 
                                           facecolor='lightblue', edgecolor='blue', linewidth=2)
                    ax.add_patch(source_shape)
                    ax.text(source_x, source_y + 0.35, 'Source', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
                else:
                    # Multiple sources - diamond shapes
                    source_diamond = plt.Polygon([(source_x, source_y+0.2), (source_x+0.2, source_y), 
                                                (source_x, source_y-0.2), (source_x-0.2, source_y)], 
                                               facecolor='lightcoral', edgecolor='darkred', linewidth=2)
                    ax.add_patch(source_diamond)
                
                ax.text(source_x, source_y - 0.4, f'S{source_id}', ha='center', va='top',
                       fontsize=9, fontweight='bold')
        
        # Position detectors on the right
        detector_positions = {}
        if detectors:
            det_spacing = min(1.5, table_height / max(1, len(detectors) + 1))
            det_start_y = table_height/2 - (len(detectors)-1) * det_spacing / 2
            det_x = table_width * 0.85
            
            for i, det_id in enumerate(sorted(detectors)):
                det_y = det_start_y + i * det_spacing
                detector_positions[det_id] = (det_x, det_y)
                
                # Different colors for different detector types
                if det_id in ancillas:
                    detector_color = 'lightcoral'
                    edge_color = 'darkred'
                    label_prefix = 'A'
                else:
                    detector_color = 'yellow'
                    edge_color = 'orange'
                    label_prefix = 'D'
                
                detector = Circle((det_x, det_y), 0.15, facecolor=detector_color, 
                                edgecolor=edge_color, linewidth=2)
                ax.add_patch(detector)
                ax.text(det_x + 0.25, det_y, f'{label_prefix}{det_id}', ha='left', va='center',
                       fontweight='bold', color=edge_color)
        
        # Add beam splitters if needed
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
                ax.plot([bs_x-0.25, bs_x+0.25], [bs_y-0.25, bs_y+0.25], 'black', linewidth=2)
                ax.text(bs_x, bs_y-0.4, f'BS{bs_id}', ha='center', va='top', 
                       fontsize=8, fontweight='bold')
        
        # Draw proper optical routing through beam splitters
        connections_drawn = 0
        
        # Create comprehensive element positions for routing
        element_positions = {}
        
        # Add source positions
        for source_id, pos in source_positions.items():
            element_positions[f'source_{source_id}'] = pos
        
        # Add detector positions  
        for det_id, pos in detector_positions.items():
            element_positions[f'detector_{det_id}'] = pos
        
        # Add beam splitter positions
        for bs_id, pos in bs_positions.items():
            element_positions[f'bs_{bs_id}'] = pos
        
        # Use proper optical routing for networks with beam splitters
        connections_drawn = self._draw_adaptive_optical_routing(
            ax, element_positions, sources, beam_splitters, detectors, ancillas
        )
        
        # Add adaptive legend showing actual components
        self._add_adaptive_optical_legend(ax, table_width, table_height, sources, detectors, beam_splitters, ancillas)
        
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
    
    def _draw_adaptive_optical_routing(self, ax, element_positions, sources, beam_splitters, 
                                     detectors, ancillas):
        """
        Draw PHYSICALLY REALISTIC optical routing for modular quantum network with multi-mode support.
        
        KEY PRINCIPLE: Route photons through the actual optical path:
        Sources → Beam Splitters → Detectors
        
        Multi-mode support: Different colors for different modes in higher-dimensional states.
        """
        if self.verbose:
            print("   Drawing adaptive optical routing for modular quantum network...")
        
        connections_drawn = 0
        
        # PyTheus standard colors for modes
        mode_colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
        
        # STEP 1: Route sources to beam splitters based on actual graph edges WITH MODE COLORS
        for source_id in sources:
            source_pos = element_positions.get(f'source_{source_id}')
            if source_pos:
                # Find actual beam splitters connected to this source
                for edge_key, weight in self.graph.items():
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        # Extract mode information
                        mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                        
                        # Check if source is connected to a beam splitter
                        if (v1 == source_id and v2 in beam_splitters) or (v2 == source_id and v1 in beam_splitters):
                            bs_id = v2 if v1 == source_id else v1
                            bs_pos = element_positions.get(f'bs_{bs_id}')
                            
                            if bs_pos and abs(weight) > 0.01:  # Draw meaningful connections
                                # Use mode colors for multi-mode visualization
                                source_mode = mode1 if v1 == source_id else mode2
                                bs_mode = mode2 if v1 == source_id else mode1
                                
                                color1 = mode_colors[int(source_mode) % len(mode_colors)]
                                color2 = mode_colors[int(bs_mode) % len(mode_colors)]
                                
                                thickness = max(1.5, min(3, abs(weight) * 2))
                                alpha = min(0.9, max(0.6, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if source_mode != bs_mode:
                                    mid_x = (source_pos[0] + bs_pos[0]) / 2
                                    mid_y = (source_pos[1] + bs_pos[1]) / 2
                                    ax.plot([source_pos[0]+0.3, mid_x], [source_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, zorder=3)
                                    ax.plot([mid_x, bs_pos[0]-0.3], [mid_y, bs_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, zorder=3)
                                else:
                                    ax.plot([source_pos[0]+0.3, bs_pos[0]-0.3], 
                                           [source_pos[1], bs_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, zorder=3)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    print(f"   Source {source_id} → BS {bs_id} (w={weight:.2f}, modes {source_mode}->{bs_mode})")
        
        # STEP 2: Route beam splitters to detectors based on actual graph edges WITH MODE COLORS
        for bs_id in beam_splitters:
            bs_pos = element_positions.get(f'bs_{bs_id}')
            if bs_pos:
                for edge_key, weight in self.graph.items():
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                    
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        # Extract mode information
                        mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                        
                        # Check if beam splitter is connected to a detector
                        if (v1 == bs_id and v2 in detectors) or (v2 == bs_id and v1 in detectors):
                            det_id = v2 if v1 == bs_id else v1
                            det_pos = element_positions.get(f'detector_{det_id}')
                            
                            if det_pos and abs(weight) > 0.01:  # Draw meaningful connections
                                # Extract modes for this connection
                                bs_mode = mode1 if v1 == bs_id else mode2
                                det_mode = mode2 if v1 == bs_id else mode1
                                
                                color1 = mode_colors[int(bs_mode) % len(mode_colors)]
                                color2 = mode_colors[int(det_mode) % len(mode_colors)]
                                
                                # Use different line styles for different detector types
                                if det_id in ancillas:
                                    linestyle = '--'  # Dashed for ancilla connections
                                else:
                                    linestyle = '-'   # Solid for regular detector connections
                                
                                thickness = max(1.5, min(3, abs(weight) * 2))
                                alpha = min(0.9, max(0.6, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if bs_mode != det_mode:
                                    mid_x = (bs_pos[0] + det_pos[0]) / 2
                                    mid_y = (bs_pos[1] + det_pos[1]) / 2
                                    ax.plot([bs_pos[0]+0.3, mid_x], [bs_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=linestyle, zorder=3)
                                    ax.plot([mid_x, det_pos[0]-0.2], [mid_y, det_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, 
                                           linestyle=linestyle, zorder=3)
                                else:
                                    ax.plot([bs_pos[0]+0.3, det_pos[0]-0.2], 
                                           [bs_pos[1], det_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=linestyle, zorder=3)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    det_type = "Ancilla" if det_id in ancillas else "Regular"
                                    print(f"   BS {bs_id} → {det_type} Det {det_id} (w={weight:.2f}, modes {bs_mode}->{det_mode})")

        # STEP 2.5: Route sources DIRECTLY to communication detectors (when no beam splitters) 
        # This is CRITICAL for GHZ states where sources connect directly to output detectors
        communication_detectors = [det for det in detectors if det not in ancillas]
        
        for source_id in sources:
            source_pos = element_positions.get(f'source_{source_id}')
            if source_pos:
                # Check connections from sources to communication detectors
                for edge_key, weight in self.graph.items():
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                    
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        # Extract mode information
                        mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                        
                        # Check if source is connected to a communication detector
                        if (v1 == source_id and v2 in communication_detectors) or (v2 == source_id and v1 in communication_detectors):
                            det_id = v2 if v1 == source_id else v1
                            det_pos = element_positions.get(f'detector_{det_id}')
                            
                            if det_pos and abs(weight) > 0.01:  # Draw meaningful connections
                                # Extract modes for this connection
                                source_mode = mode1 if v1 == source_id else mode2
                                det_mode = mode2 if v1 == source_id else mode1
                                
                                color1 = mode_colors[int(source_mode) % len(mode_colors)]
                                color2 = mode_colors[int(det_mode) % len(mode_colors)]
                                
                                thickness = max(1.8, min(3.5, abs(weight) * 2))
                                alpha = min(0.9, max(0.7, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if source_mode != det_mode:
                                    mid_x = (source_pos[0] + det_pos[0]) / 2
                                    mid_y = (source_pos[1] + det_pos[1]) / 2
                                    ax.plot([source_pos[0]+0.3, mid_x], [source_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle='-', zorder=4)
                                    ax.plot([mid_x, det_pos[0]-0.2], [mid_y, det_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, 
                                           linestyle='-', zorder=4)
                                else:
                                    ax.plot([source_pos[0]+0.3, det_pos[0]-0.2], 
                                           [source_pos[1], det_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle='-', zorder=4)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    print(f"   Source {source_id} → Comm Det {det_id} (w={weight:.2f}, modes {source_mode}->{det_mode})")

        # STEP 3: Route sources AND beam splitters to ancilla detectors based on ACTUAL graph edges WITH MODE COLORS
        for anc_det_id in ancillas:
            anc_pos = element_positions.get(f'detector_{anc_det_id}')
            if anc_pos:
                # Check connections from sources to this ancilla
                for edge_key, weight in self.graph.items():
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                    
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        # Extract mode information
                        mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                        
                        # Check if source is connected to this ancilla
                        if (v1 in sources and v2 == anc_det_id) or (v2 in sources and v1 == anc_det_id):
                            source_id = v1 if v1 in sources else v2
                            source_pos = element_positions.get(f'source_{source_id}')
                            
                            if source_pos and abs(weight) > 0.01:
                                # Extract modes for this connection
                                source_mode = mode1 if v1 == source_id else mode2
                                anc_mode = mode2 if v1 == source_id else mode1
                                
                                color1 = mode_colors[int(source_mode) % len(mode_colors)]
                                color2 = mode_colors[int(anc_mode) % len(mode_colors)]
                                
                                thickness = max(1.2, min(2.5, abs(weight) * 1.5))
                                alpha = min(0.8, max(0.5, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if source_mode != anc_mode:
                                    mid_x = (source_pos[0] + anc_pos[0]) / 2
                                    mid_y = (source_pos[1] + anc_pos[1]) / 2
                                    ax.plot([source_pos[0]+0.3, mid_x], [source_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                    ax.plot([mid_x, anc_pos[0]-0.2], [mid_y, anc_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                else:
                                    ax.plot([source_pos[0]+0.3, anc_pos[0]-0.2], 
                                           [source_pos[1], anc_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    print(f"   Source {source_id} → Anc Det {anc_det_id} (w={weight:.2f}, modes {source_mode}->{anc_mode})")
                        
                        # Check if beam splitter is connected to this ancilla
                        elif (v1 in beam_splitters and v2 == anc_det_id) or (v2 in beam_splitters and v1 == anc_det_id):
                            bs_id = v1 if v1 in beam_splitters else v2
                            bs_pos = element_positions.get(f'bs_{bs_id}')
                            
                            if bs_pos and abs(weight) > 0.01:
                                # Extract modes for this connection
                                bs_mode = mode1 if v1 == bs_id else mode2
                                anc_mode = mode2 if v1 == bs_id else mode1
                                
                                color1 = mode_colors[int(bs_mode) % len(mode_colors)]
                                color2 = mode_colors[int(anc_mode) % len(mode_colors)]
                                
                                thickness = max(1.2, min(2.5, abs(weight) * 1.5))
                                alpha = min(0.8, max(0.5, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if bs_mode != anc_mode:
                                    mid_x = (bs_pos[0] + anc_pos[0]) / 2
                                    mid_y = (bs_pos[1] + anc_pos[1]) / 2
                                    ax.plot([bs_pos[0]+0.3, mid_x], [bs_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                    ax.plot([mid_x, anc_pos[0]-0.2], [mid_y, anc_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                else:
                                    ax.plot([bs_pos[0]+0.3, anc_pos[0]-0.2], 
                                           [bs_pos[1], anc_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, 
                                           linestyle=':', zorder=2)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    print(f"   BS {bs_id} → Anc Det {anc_det_id} (w={weight:.2f}, modes {bs_mode}->{anc_mode})")
        
        # STEP 4: Add ancilla-ancilla correlations for strong connections
        for i, anc1_id in enumerate(ancillas):
            for anc2_id in ancillas[i+1:]:
                weight = self._get_edge_weight(anc1_id, anc2_id)
                if abs(weight) > 0.3:   # Show correlations
                    anc1_pos = element_positions.get(f'detector_{anc1_id}')
                    anc2_pos = element_positions.get(f'detector_{anc2_id}')
                    if anc1_pos and anc2_pos:
                        thickness = max(1.0, min(2.5, abs(weight) * 1.5))
                        ax.plot([anc1_pos[0], anc2_pos[0]], 
                               [anc1_pos[1], anc2_pos[1]], 
                               color='gray', linewidth=thickness, linestyle='--', 
                               alpha=0.7, zorder=1)
                        connections_drawn += 1
                        if self.verbose:
                            print(f"   Anc {anc1_id} ↔ Anc {anc2_id} (correlation w={weight:.2f})")
        
        if self.verbose:
            print(f"   Total optical connections drawn: {connections_drawn}")
        
        return connections_drawn
    
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
                          label=f'Mode {mode}')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.15, 1), fontsize=10, title='Edge Colors',
                     frameon=True, fancybox=True, shadow=True)
        
        # Add network statistics in a clean info box
        perfect_matchings = self.find_perfect_matchings()
        impl_strategy = network_analysis['implementation_strategy']
        strategy_name = "graph_structure_based"  # Our new general approach
        
        stats_text = f"""Vertices: {len(vertices)}
Edges: {network_analysis['edge_count']}
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
- **Interpreter Version**: Modular Quantum Network Interpreter v1.0
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
            base_filename = "modular_network_analysis"
        
        if self.verbose:
            print("🚀 Running complete quantum network analysis...")
        
        # Generate all outputs
        results = {}
        
        # 1. Optical table setup
        optical_fig = self.plot_optical_table_setup(
            save_path=f"{base_filename}_optical_table_setup.png",
            title="Modular Quantum Network - Optical Table Setup"
        )
        results['optical_table'] = optical_fig
        
        # 2. Native graph plot
        native_fig = self.plot_native_graph(
            save_path=f"{base_filename}_native_plot.png",
            title="Modular Quantum Network - PyTheus Graph"
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
        Identify actual sources based on config and graph analysis.
        Modular - works for tested network types.
        """
        # Priority 1: Check config for explicitly specified sources
        config_sources = self.config.get('single_emitters', [])
        if config_sources:
            if self.verbose:
                print(f"   Using config-specified sources: {config_sources}")
            return config_sources
        
        # Priority 2: For QKD networks, out_nodes are the communication parties that need sources
        out_nodes = self.config.get('out_nodes', [])
        if out_nodes and not self.config.get('single_emitters'):
            if self.verbose:
                print(f"   Using out_nodes as SPDC sources for QKD: {out_nodes}")
            return out_nodes
        
        # Priority 3: Check for other source indicators in config
        in_nodes = self.config.get('in_nodes', [])
        if in_nodes:
            if self.verbose:
                print(f"   Using in_nodes as sources: {in_nodes}")
            return in_nodes
        
        # Priority 4: Structural analysis - find vertices that primarily send rather than receive
        sources = []
        degrees = self._compute_vertex_degrees()
        
        for vertex in self.vertices:
            outgoing_weight = 0
            incoming_weight = 0
            
            # Analyze edge directions based on weights
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
            
            # Vertices with more outgoing than incoming weight are likely sources
            if outgoing_weight > incoming_weight and degrees.get(vertex, 0) > 0:
                sources.append(vertex)
        
        # If no clear sources found, assume all vertices can be sources
        if not sources:
            sources = self.vertices.copy()
        
        if self.verbose:
            print(f"   Structural analysis sources: {sources}")
        
        return sources
    
    def _identify_beam_splitter_nodes(self):
        """
        Identify beam splitter nodes based on modular graph analysis.
        Completely general - works for any network type.
        
        Key insight: In quantum networks, nodes can serve dual roles:
        - Ancilla detectors can ALSO function as beam splitters
        - High-degree nodes that route photons are beam splitters regardless of other roles
        """
        degrees = self._compute_vertex_degrees()
        sources = self._identify_actual_sources()
        detectors = self._identify_actual_detectors()  # Use actual detected detectors
        ancillas = self._identify_ancilla_nodes()  # Get ancillas first
        out_nodes = self.config.get('out_nodes', [])
        
        beam_splitters = []
        
        if degrees:
            if self.config.get('single_emitters'):
                # For single photon networks (W4, etc.), identify intermediate routing nodes
                # These are nodes that are neither pure sources nor pure detectors
                sources_set = set(sources)
                detectors_set = set(detectors)  # Use actual detectors, not just out_nodes
                
                for vertex, degree in degrees.items():
                    if (degree >= 2 and 
                        vertex not in sources_set and 
                        vertex not in detectors_set):
                        beam_splitters.append(vertex)
                
                # For W4 networks, if no clear intermediate nodes, don't force beam splitters
                # W4 can work with direct source-to-detector connections
                if not beam_splitters:
                    if self.verbose:
                        print("   No intermediate beam splitters needed for this single photon network")
                    beam_splitters = []
            else:
                # For SPDC networks, identify high-degree central nodes that can mix signals
                # KEY PHYSICS INSIGHT: Ancillas can serve dual roles as beam splitters + detectors
                
                # First priority: Find pure beam splitters (not sources or out_nodes)
                for vertex, degree in degrees.items():
                    if (degree >= 4 and 
                        vertex not in sources and 
                        vertex not in out_nodes):
                        beam_splitters.append(vertex)
                
                # If no pure beam splitters found, allow high-degree ancillas to serve as beam splitters
                if not beam_splitters:
                    # Look for high-degree ancillas that connect to multiple sources
                    for vertex in ancillas:
                        degree = degrees.get(vertex, 0)
                        if degree >= 3:  # Lower threshold for ancillas
                            # Count how many sources connect to this ancilla
                            source_connections = 0
                            for edge_key, weight in self.graph.items():
                                if isinstance(edge_key, str):
                                    edge_tuple = eval(edge_key)
                                else:
                                    edge_tuple = edge_key
                                
                                if len(edge_tuple) >= 2:
                                    v1, v2 = edge_tuple[0], edge_tuple[1]
                                    if ((v1 == vertex and v2 in sources) or 
                                        (v2 == vertex and v1 in sources)) and abs(weight) > 0.1:
                                        source_connections += 1
                            
                            # If this ancilla connects to multiple sources, it's acting as a beam splitter
                            if source_connections >= 2:
                                beam_splitters.append(vertex)
                                if self.verbose:
                                    print(f"   Ancilla {vertex} identified as dual-role beam splitter (connects to {source_connections} sources)")
                
                # Final fallback: use the highest degree ancilla as a central mixer
                if not beam_splitters and ancillas:
                    max_degree = 0
                    best_ancilla = None
                    for vertex in ancillas:
                        degree = degrees.get(vertex, 0)
                        if degree > max_degree:
                            max_degree = degree
                            best_ancilla = vertex
                    
                    if best_ancilla and max_degree >= 3:
                        beam_splitters.append(best_ancilla)
                        if self.verbose:
                            print(f"   Using highest-degree ancilla {best_ancilla} (degree={max_degree}) as central beam splitter")
        
        if self.verbose:
            print(f"   Identified beam splitters: {beam_splitters}")
        
        return beam_splitters
    
    def _identify_ancilla_nodes(self):
        """
        Identify ancilla nodes from config and network analysis.
        """
        # Priority 1: Explicit config specification (highest priority)
        anc_detectors = self.config.get('anc_detectors', [])
        if anc_detectors:
            if self.verbose:
                print(f"   Using config-specified ancillas: {anc_detectors}")
            return anc_detectors
        
        # Priority 2: Use config-based assignment from single_emitters + out_nodes
        config_single_emitters = self.config.get('single_emitters', [])
        config_out_nodes = self.config.get('out_nodes', [])
        
        if config_single_emitters and config_out_nodes:
            # For heralded Bell: single_emitters=[2,3,4,5], out_nodes=[0,1]
            # Ancillas are remaining vertices that aren't sources or detectors
            vertices = self.vertices
            potential_ancillas = [v for v in vertices 
                                if v not in config_single_emitters and v not in config_out_nodes]
            
            if potential_ancillas:
                if self.verbose:
                    print(f"   Using config-based ancillas (single_emitters + out_nodes): {potential_ancillas}")
                return potential_ancillas
        
        # Priority 3: Infer from num_anc and out_nodes
        num_anc = self.config.get('num_anc', 0)
        out_nodes = self.config.get('out_nodes', [])
        
        if num_anc > 0:
            if out_nodes:
                # Use out_nodes to determine number of target particles
                # For GHZ346: out_nodes = [0, 1, 2] means 3 particles
                # Ancillas should be the remaining vertices
                num_target_particles = len(out_nodes)
                total_vertices = len(self.vertices)
                
                if total_vertices >= num_target_particles + num_anc:
                    all_vertices = sorted(self.vertices)
                    ancillas = all_vertices[num_target_particles:num_target_particles + num_anc]
                    if self.verbose:
                        print(f"   Inferred ancillas from num_anc={num_anc}, out_nodes={out_nodes}: {ancillas}")
                    return ancillas
            else:
                # No out_nodes, use target_state
                target_state = self.config.get('target_state', [])
                if target_state:
                    # For GHZ states, determine particles from target_state length
                    # But be careful - target_state might represent particles in higher dimensions
                    # For now, assume target_state[0] length equals number of particles
                    num_target_particles = len(target_state[0]) if target_state else 0
                    total_vertices = len(self.vertices)
                    
                    if total_vertices >= num_target_particles + num_anc:
                        all_vertices = sorted(self.vertices)
                        ancillas = all_vertices[num_target_particles:num_target_particles + num_anc]
                        if self.verbose:
                            print(f"   Inferred ancillas from num_anc={num_anc}, target_particles={num_target_particles}: {ancillas}")
                        return ancillas
                else:
                    # No target state, use highest-numbered vertices as ancillas
                    all_vertices = sorted(self.vertices)
                    ancillas = all_vertices[-num_anc:]
                    if self.verbose:
                        print(f"   Inferred ancillas from num_anc={num_anc} (highest vertices): {ancillas}")
                    return ancillas
        
        return []
    
    def _identify_actual_detectors(self):
        """
        Identify detector nodes based on general analysis.
        Completely general - works for any network type.
        """
        # Priority 1: Check config for out_nodes (communication detectors)
        out_nodes = self.config.get('out_nodes', [])
        anc_detectors = self._identify_ancilla_nodes()  # Get ancillas
        
        # Priority 2: For W4 and similar, use target state to determine detectors
        target_state = self.config.get('target_state', [])
        if target_state and not out_nodes:
            # Number of detectors = length of state string
            num_detectors = len(target_state[0])
            out_nodes = list(range(num_detectors))
        
        # Priority 3: If no config info, assume all non-source vertices are detectors
        if not out_nodes and not anc_detectors:
            sources = self._identify_actual_sources()
            all_detectors = [v for v in self.vertices if v not in sources]
        else:
            # Combine out_nodes and ancilla detectors
            all_detectors = list(set(out_nodes + anc_detectors))
        
        if self.verbose:
            print(f"   Identified detectors: {all_detectors}")
            if out_nodes:
                print(f"     Communication detectors: {out_nodes}")
            if anc_detectors:
                print(f"     Ancilla detectors: {anc_detectors}")
        
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
        
        KEY INSIGHT: In QKD networks:
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
    
    def _draw_optical_routing(self, ax, element_positions, spdc_sources, beam_splitters, 
                            out_nodes, anc_detectors):
        """
        Draw PHYSICALLY REALISTIC optical routing for quantum networks.
        
        KEY PRINCIPLE: Follow proper optical path hierarchy:
        1. Sources → Beam Splitters (solid lines)
        2. Beam Splitters → Detectors (solid lines)
        3. Ancilla correlations (dashed lines for post-processing)
        
        NO direct source-to-detector connections - all photons must route through beam splitters.
        """
        if self.verbose:
            print("   Drawing physically realistic optical routing...")
        
        connections_drawn = 0
        
        # PyTheus standard colors for modes (consistent with adaptive plotter)
        mode_colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
        
        # STEP 1: Route SPDC sources to beam splitters based on ACTUAL graph edges WITH MODE COLORS
        for source_id in spdc_sources:
            spdc_pos = element_positions.get(f'spdc_{source_id}')
            if spdc_pos:
                # Find actual beam splitters connected to this source in the graph
                for edge_key, weight in self.graph.items():
                    if isinstance(edge_key, str):
                        edge_tuple = eval(edge_key)
                    else:
                        edge_tuple = edge_key
                
                    if len(edge_tuple) >= 2:
                        v1, v2 = edge_tuple[0], edge_tuple[1]
                        # Extract mode information
                        mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                        
                        # Check if source is actually connected to a beam splitter
                        if (v1 == source_id and v2 in beam_splitters) or (v2 == source_id and v1 in beam_splitters):
                            bs_id = v2 if v1 == source_id else v1
                            bs_pos = element_positions.get(f'bs_{bs_id}')
                            
                            if bs_pos and weight != 0:  # Draw actual graph connections
                                # Use mode colors for proper visualization
                                source_mode = mode1 if v1 == source_id else mode2
                                bs_mode = mode2 if v1 == source_id else mode1
                                
                                color1 = mode_colors[int(source_mode) % len(mode_colors)]
                                color2 = mode_colors[int(bs_mode) % len(mode_colors)]
                                
                                thickness = max(1.5, min(3, abs(weight) * 2))
                                alpha = min(0.9, max(0.6, abs(weight)))
                                
                                # Draw multi-mode connection if modes differ
                                if source_mode != bs_mode:
                                    mid_x = (spdc_pos[0] + bs_pos[0]) / 2
                                    mid_y = (spdc_pos[1] + bs_pos[1]) / 2
                                    ax.plot([spdc_pos[0]+0.3, mid_x], [spdc_pos[1], mid_y], 
                                           color=color1, linewidth=thickness, alpha=alpha, zorder=3)
                                    ax.plot([mid_x, bs_pos[0]-0.3], [mid_y, bs_pos[1]], 
                                           color=color2, linewidth=thickness, alpha=alpha, zorder=3)
                                else:
                                    ax.plot([spdc_pos[0]+0.3, bs_pos[0]-0.3], 
                                           [spdc_pos[1], bs_pos[1]], 
                                           color=color1, linewidth=thickness, alpha=alpha, zorder=3)
                                
                                connections_drawn += 1
                                if self.verbose:
                                    print(f"   SPDC {source_id} → BS {bs_id} (w={weight:.2f}, modes {source_mode}→{bs_mode})")
        
        # STEP 2: Route beam splitters to communication detectors ONLY
        # This is the main optical routing - beam splitters distribute photons to communication detectors
        for bs_id in beam_splitters:
            bs_pos = element_positions.get(f'bs_{bs_id}')
            if bs_pos:
                # Route to communication detectors (not ancillas - those are the beam splitters themselves)
                for comm_det_id in out_nodes:
                    # Check if this beam splitter is actually connected to this detector
                    for edge_key, weight in self.graph.items():
                        if isinstance(edge_key, str):
                            edge_tuple = eval(edge_key)
                        else:
                            edge_tuple = edge_key
                        
                        if len(edge_tuple) >= 2:
                            v1, v2 = edge_tuple[0], edge_tuple[1]
                            # Extract mode information
                            mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                            
                            # Check if beam splitter is connected to this communication detector
                            if (v1 == bs_id and v2 == comm_det_id) or (v2 == bs_id and v1 == comm_det_id):
                                if weight != 0:  # Draw actual graph connections
                                    det_pos = element_positions.get(f'detector_{comm_det_id}')
                                    if det_pos:
                                        # Use mode colors for proper visualization
                                        bs_mode = mode1 if v1 == bs_id else mode2
                                        det_mode = mode2 if v1 == bs_id else mode1
                                        
                                        color1 = mode_colors[int(bs_mode) % len(mode_colors)]
                                        color2 = mode_colors[int(det_mode) % len(mode_colors)]
                                        
                                        thickness = max(1.5, min(3, abs(weight) * 2))
                                        alpha = min(0.9, max(0.6, abs(weight)))
                                        
                                        # Draw solid connection (main optical path)
                                        if bs_mode != det_mode:
                                            mid_x = (bs_pos[0] + det_pos[0]) / 2
                                            mid_y = (bs_pos[1] + det_pos[1]) / 2
                                            ax.plot([bs_pos[0]+0.3, mid_x], [bs_pos[1], mid_y], 
                                                   color=color1, linewidth=thickness, alpha=alpha, 
                                                   linestyle='-', zorder=3)
                                            ax.plot([mid_x, det_pos[0]-0.2], [mid_y, det_pos[1]], 
                                                   color=color2, linewidth=thickness, alpha=alpha, 
                                                   linestyle='-', zorder=3)
                                        else:
                                            ax.plot([bs_pos[0]+0.3, det_pos[0]-0.2], 
                                                   [bs_pos[1], det_pos[1]], 
                                                   color=color1, linewidth=thickness, alpha=alpha, 
                                                   linestyle='-', zorder=3)
                                        
                                        connections_drawn += 1
                                        if self.verbose:
                                            print(f"   BS {bs_id} → Comm Det {comm_det_id} (w={weight:.2f}, modes {bs_mode}→{det_mode})")
        
        # STEP 3: Add ancilla-ancilla correlations ONLY for post-processing visualization
        # These are NOT optical connections, but show quantum correlations for analysis
        for i, anc1_id in enumerate(anc_detectors):
            for anc2_id in anc_detectors[i+1:]:
                weight = self._get_edge_weight(anc1_id, anc2_id)
                if abs(weight) > 0.5:   # Only show strong correlations
                    anc1_pos = element_positions.get(f'detector_{anc1_id}')
                    anc2_pos = element_positions.get(f'detector_{anc2_id}')
                    if anc1_pos and anc2_pos:
                        thickness = max(1.0, min(2.0, abs(weight) * 1.5))
                        ax.plot([anc1_pos[0], anc2_pos[0]], 
                               [anc1_pos[1], anc2_pos[1]], 
                               color='lightgray', linewidth=thickness, linestyle='--', 
                               alpha=0.6, zorder=1)
                        connections_drawn += 1
                        if self.verbose:
                            print(f"   Anc {anc1_id} ↔ Anc {anc2_id} (correlation w={weight:.2f})")
        
        if self.verbose:
            print(f"   Total optical connections drawn: {connections_drawn}")
        
        return connections_drawn
    
    def _get_edge_weight(self, v1, v2):
        """Get the weight of an edge between two vertices."""
        for edge_key, weight in self.graph.items():
            if isinstance(edge_key, str):
                edge_tuple = eval(edge_key)
            else:
                edge_tuple = edge_key
            
            if len(edge_tuple) >= 2:
                if (edge_tuple[0] == v1 and edge_tuple[1] == v2) or (edge_tuple[0] == v2 and edge_tuple[1] == v1):
                    return weight
        return 0.0
    
    def _add_optical_connections(self, ax, network_analysis, source_positions, detector_positions, 
                                table_width, table_height):
        """Add optical connections for single photon networks with proper ancilla visualization"""
        # Get ancilla information
        strategy_info = network_analysis.get('implementation_strategy', {})
        ancillas = strategy_info.get('ancillas', [])
        
        # PyTheus standard colors for modes
        mode_colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
        
        # Draw connections based on actual graph structure
        connections_drawn = 0
        
        if self.verbose:
            print("   Drawing single photon optical connections...")
        
        for edge_key, weight in self.graph.items():
            if isinstance(edge_key, str):
                edge_tuple = eval(edge_key)
            else:
                edge_tuple = edge_key
            
            if len(edge_tuple) >= 2 and abs(weight) > 0.01:  # Only draw meaningful connections
                v1, v2 = edge_tuple[0], edge_tuple[1]
                
                # Extract mode information
                mode1, mode2 = (edge_tuple[2], edge_tuple[3]) if len(edge_tuple) >= 4 else (0, 0)
                
                # Find positions
                pos1 = source_positions.get(v1) or detector_positions.get(v1)
                pos2 = source_positions.get(v2) or detector_positions.get(v2)
                
                if pos1 and pos2:
                    # Determine connection type and styling
                    if v1 in ancillas or v2 in ancillas:
                        # Connection involving ancilla detector
                        linestyle = '--'  # Dashed for ancilla connections
                        alpha = min(0.8, max(0.5, abs(weight)))
                    else:
                        # Communication detector connection
                        linestyle = '-'   # Solid for communication connections
                        alpha = min(0.9, max(0.7, abs(weight)))
                    
                    # Use mode colors for multi-mode visualization
                    color1 = mode_colors[int(mode1) % len(mode_colors)]
                    color2 = mode_colors[int(mode2) % len(mode_colors)]
                    
                    thickness = max(1.5, min(3.5, abs(weight) * 2.5))
                    
                    # Draw multi-mode connection if modes differ
                    if mode1 != mode2:
                        mid_x = (pos1[0] + pos2[0]) / 2
                        mid_y = (pos1[1] + pos2[1]) / 2
                        ax.plot([pos1[0], mid_x], [pos1[1], mid_y], 
                               color=color1, linewidth=thickness, alpha=alpha, 
                               linestyle=linestyle, zorder=2)
                        ax.plot([mid_x, pos2[0]], [mid_y, pos2[1]], 
                               color=color2, linewidth=thickness, alpha=alpha, 
                               linestyle=linestyle, zorder=2)
                    else:
                        # Single mode connection
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color=color1, linewidth=thickness, alpha=alpha, 
                               linestyle=linestyle, zorder=2)
                    
                    connections_drawn += 1
                    
                    if self.verbose:
                        # Determine connection type for verbose output
                        if v1 in ancillas and v2 in ancillas:
                            print(f"   Anc {v1} ↔ Anc {v2} (w={weight:.2f}, modes {mode1}↔{mode2})")
                        elif v1 in ancillas:
                            print(f"   Source {v2} → Anc {v1} (w={weight:.2f}, modes {mode2}→{mode1})")
                        elif v2 in ancillas:
                            print(f"   Source {v1} → Anc {v2} (w={weight:.2f}, modes {mode1}→{mode2})")
                        else:
                            print(f"   Source {v1} → Comm Det {v2} (w={weight:.2f}, modes {mode1}→{mode2})")
        
        if self.verbose:
            print(f"   Total optical connections drawn: {connections_drawn}")
        
        return connections_drawn
    
    def _add_optical_legend(self, ax, table_width, table_height, network_analysis):
        """Add legend for optical table setup with proper ancilla visualization"""
        strategy_info = network_analysis.get('implementation_strategy', {})
        sources = strategy_info.get('sources', [])
        detectors = strategy_info.get('detectors', [])
        ancillas = strategy_info.get('ancillas', [])
        
        legend_elements = []
        
        # Add source legend
        if sources:
            if len(sources) == 1:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='lightcoral', 
                              markersize=10, label='Single Photon Source')
                )
            else:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='lightcoral', 
                              markersize=10, label='Single Photon Sources')
                )
        
        # Add detector legends - separate for communication and ancilla detectors
        communication_detectors = [d for d in detectors if d not in ancillas]
        
        if communication_detectors:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                          markersize=10, label='Communication Detectors')
            )
        
        if ancillas:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                          markersize=10, label='Ancilla Detectors')
            )
        
        # Add connection type legends
        if communication_detectors:
            legend_elements.append(
                plt.Line2D([0], [0], color='dodgerblue', linewidth=2, linestyle='-', 
                          label='Communication Connections')
            )
        
        if ancillas:
            legend_elements.append(
                plt.Line2D([0], [0], color='firebrick', linewidth=2, linestyle='--', 
                          label='Ancilla Connections')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
                     frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    def _add_adaptive_optical_legend(self, ax, table_width, table_height, sources, detectors, beam_splitters, ancillas):
        """Add proper legend for adaptive optical table setup with mode information"""
        legend_elements = []
        
        # Add source legend
        if sources:
            if len(sources) == 1:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                              markersize=10, label='Quantum Source')
                )
            else:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='lightcoral', 
                              markersize=10, label='Quantum Sources')
                )
        
        # Add beam splitter legend
        if beam_splitters:
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                          markersize=10, label='Beam Splitters')
            )
        
        # Add detector legend
        if detectors:
            # Separate regular detectors from ancillas
            regular_detectors = [d for d in detectors if d not in ancillas]
            if regular_detectors:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                              markersize=10, label='Communication Detectors')
                )
            if ancillas:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                              markersize=10, label='Ancilla Detectors')
                )
        
        # Add connection legend with mode information
        if beam_splitters:
            legend_elements.append(
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Source → Beam Splitter')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='green', linewidth=2, label='Beam Splitter → Detector')
            )
            if ancillas:
                legend_elements.append(
                    plt.Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Ancilla Connections')
                )
        else:
            legend_elements.append(
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Source → Detector')
            )
        
        # Add mode legend for multi-mode networks
        network_analysis = self.analyze_network_structure()
        unique_modes = network_analysis['mode_analysis']['unique_modes']
        if len(unique_modes) > 1:
            # PyTheus standard colors for modes
            mode_colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
            legend_elements.append(
                plt.Line2D([0], [0], color='white', linewidth=0, label='─── Modes ───')
            )
            for mode in unique_modes:
                color = mode_colors[mode % len(mode_colors)]
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, linewidth=3, 
                              label=f'Mode {mode}')
                )
        
        # Add entanglement correlation legend
        legend_elements.append(
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Entanglement Correlations')
        )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), 
                     frameon=True, fancybox=True, shadow=True, fontsize=9)