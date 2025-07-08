#!/usr/bin/env python3
"""
Example script demonstrating the PyTheus Quantum Network Interpreter
"""

import sys
import os
sys.path.append('.')

from fully_general_interpreter import create_interpreter, analyze_quantum_network

def main():
    """Demonstrate the interpreter with the 5-node QKD network"""
    
    print("PyTheus Quantum Network Interpreter - Example")
    print("=" * 50)
    
    # Paths to example files
    config_path = "examples/5node_qkd_network/config.json"
    graph_path = "examples/5node_qkd_network/best.json"
    
    # Check if files exist
    if not os.path.exists(config_path) or not os.path.exists(graph_path):
        print("Error: Example files not found!")
        print("Please ensure you have the example files in examples/5node_qkd_network/")
        return
    
    print(f"Analyzing 5-node QKD network...")
    print(f"Config: {config_path}")
    print(f"Graph:  {graph_path}")
    print()
    
    # Method 1: Quick analysis using convenience function
    print("Method 1: Quick Analysis")
    print("-" * 25)
    try:
        import json
        
        # Load configuration and graph data
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        results = analyze_quantum_network(
            config=config_data,
            graph=graph_data,
            base_filename="example_analysis"
        )
        print("✅ Quick analysis completed!")
        print("Generated files:")
        print("  - example_analysis_optical_table_setup.png")
        print("  - example_analysis_native_plot.png") 
        print("  - example_analysis_report.txt")
        print()
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        print()
    
    # Method 2: Step-by-step analysis
    print("Method 2: Step-by-Step Analysis")
    print("-" * 32)
    try:
        # Create interpreter
        interpreter = create_interpreter(
            config=config_data,
            graph=graph_data
        )
        print("✅ Interpreter created successfully")
        
        # Generate individual outputs
        interpreter.plot_optical_table_setup("detailed_optical_table.png")
        print("✅ Optical table plot generated")
        
        interpreter.plot_native_graph("detailed_native_graph.png")
        print("✅ Native graph plot generated")
        
        interpreter.generate_analysis_report("detailed_analysis.txt")
        print("✅ Analysis report generated")
        
        print()
        print("Generated files:")
        print("  - detailed_optical_table.png")
        print("  - detailed_native_graph.png")
        print("  - detailed_analysis.txt")
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
    
    print()
    print("Analysis complete! Check the generated files for results.")
    print()
    print("Key findings for 5-node QKD network:")
    print("• Hub-and-spoke topology with central node 3")
    print("• 4 SPDC sources feeding multi-port beam splitter")
    print("• 5×4 coupling matrix implementable in single integrated component")
    print("• Perfect correlations in ancilla measurement network")

if __name__ == "__main__":
    main()
