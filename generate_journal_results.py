#!/usr/bin/env python3
"""
Comprehensive test suite for journal article - generate all network types with latest interpreter.
Creates native graphs, optical tables, and reports for:
- 5-node QKD network
- W4 state  
- Heralded Bell state
- GHZ state
"""

import sys
import os
sys.path.insert(0, '/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus')

from modular_interpreter import ModularQuantumNetworkInterpreter

def test_5node_qkd():
    """Test 5-node QKD network with latest improvements."""
    print("🔍 Testing 5-Node QKD Network")
    print("="*50)
    
    config_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/output/5node_optimal_quantum_network/5node_optimal_network_2/config.json"
    graph_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/output/5node_optimal_quantum_network/5node_optimal_network_2/best.json"
    
    interpreter = ModularQuantumNetworkInterpreter(verbose=True)
    interpreter.load_config(config_path)
    interpreter.load_graph(graph_path)
    
    # Generate all outputs
    results = interpreter.run_complete_analysis("journal_5node_qkd")
    print("✅ 5-Node QKD analysis complete\n")
    return results

def test_w4_state():
    """Test W4 state network using actual PyTheus example."""
    print("🔍 Testing W4 State Network")
    print("="*50)
    
    config_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/W4_sp/config_W4_sp.json"
    graph_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/W4_sp/plot_W4_sp_clean-10-4-0.2000_0.0000.json"
    
    interpreter = ModularQuantumNetworkInterpreter(verbose=True)
    interpreter.load_config(config_path)
    interpreter.load_graph(graph_path)
    
    results = interpreter.run_complete_analysis("journal_w4_state")
    print("✅ W4 State analysis complete\n")
    return results

def test_heralded_bell():
    """Test heralded Bell state network using actual PyTheus example."""
    print("🔍 Testing Heralded Bell State Network") 
    print("="*50)
    
    config_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/heralded_bell_sp/config_heralded_bell_2d_sp.json"
    graph_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/heralded_bell_sp/plot_heralded_bell_sp.json"
    
    interpreter = ModularQuantumNetworkInterpreter(verbose=True)
    interpreter.load_config(config_path)
    interpreter.load_graph(graph_path)
    
    results = interpreter.run_complete_analysis("journal_heralded_bell")
    print("✅ Heralded Bell State analysis complete\n")
    return results

def test_ghz346_state():
    """Test GHZ346 state network using actual PyTheus example."""
    print("🔍 Testing GHZ346 State Network")
    print("="*50)
    
    config_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/ghz_346/config_ghz_346.json"
    graph_path = "/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus/pytheus/graphs/HighlyEntangledStates/ghz_346/plot_ghz_346_clean-17-22-0.0588_0.0000.json"
    
    interpreter = ModularQuantumNetworkInterpreter(verbose=True)
    interpreter.load_config(config_path)
    interpreter.load_graph(graph_path)
    
    results = interpreter.run_complete_analysis("journal_ghz346_state")
    print("✅ GHZ346 State analysis complete\n")
    return results

def main():
    """Run comprehensive test suite for journal article."""
    print("🚀 Comprehensive Journal Article Test Suite")
    print("="*60)
    print("Generating results for:")
    print("• 5-Node QKD Network")
    print("• W4 State") 
    print("• Heralded Bell State")
    print("• GHZ346 State")
    print("="*60)
    
    os.chdir('/home/rithvik/nvme_data2/Work-PyTheusRL/pytheus-quantum-network-interpreter')
    
    results = {}
    
    try:
        results['qkd'] = test_5node_qkd()
    except Exception as e:
        print(f"❌ QKD test failed: {e}")
    
    try:
        results['w4'] = test_w4_state()
    except Exception as e:
        print(f"❌ W4 test failed: {e}")
        
    try:
        results['bell'] = test_heralded_bell()
    except Exception as e:
        print(f"❌ Bell test failed: {e}")
        
    try:
        results['ghz346'] = test_ghz346_state()
    except Exception as e:
        print(f"❌ GHZ346 test failed: {e}")
    
    print("🎉 Comprehensive test suite complete!")
    print("\nGenerated files:")
    print("📊 Native graphs: journal_*_native_plot.png")
    print("🔬 Optical tables: journal_*_optical_table_setup.png") 
    print("📄 Reports: journal_*_report.txt")
    
    return results

if __name__ == "__main__":
    main()
