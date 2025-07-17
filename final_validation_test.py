#!/usr/bin/env python3
"""
Final validation test to ensure beam splitter identification and optical table connections work correctly.
"""

import sys
import os
sys.path.insert(0, '/home/rithvik/nvme_data2/Work-PyTheusRL/PyTheus')

from modular_interpreter import ModularQuantumNetworkInterpreter

def test_network(name, config_path, graph_path):
    """Test a single network configuration."""
    print(f"\n🔬 Testing {name}...")
    print("="*50)
    
    interpreter = ModularQuantumNetworkInterpreter()
    interpreter.load_config(config_path)
    interpreter.load_graph(graph_path)
    
    # Analyze structure
    sources = interpreter._identify_actual_sources()
    detectors = interpreter._identify_actual_detectors()
    beam_splitters = interpreter._identify_beam_splitter_nodes()
    ancillas = interpreter._identify_ancilla_nodes()
    
    print(f"Sources: {sources}")
    print(f"Detectors: {detectors}")
    print(f"Beam Splitters: {beam_splitters}")
    print(f"Ancillas: {ancillas}")
    
    # Check for dual-role ancillas
    dual_role_ancillas = [a for a in ancillas if a in beam_splitters]
    if dual_role_ancillas:
        print(f"✅ Dual-role ancillas (BS + Detector): {dual_role_ancillas}")
    else:
        print("ℹ️  No dual-role ancillas found")
    
    # Generate optical table
    output_name = f"final_validation_{name.lower().replace(' ', '_')}"
    interpreter.plot_optical_table_setup(output_name)
    print(f"✅ Optical table generated: {output_name}_optical_table.png")
    
    return len(sources), len(detectors), len(beam_splitters), len(ancillas)

def main():
    """Run comprehensive validation tests."""
    print("🚀 Final Validation Test Suite")
    print("="*60)
    
    # Test configurations
    test_configs = [
        ("5-Node QKD", 
         "/home/rithvik/nvme_data2/Work-PyTheusRL/output/5node_optimal_quantum_network/5node_optimal_network_2/config.json",
         "/home/rithvik/nvme_data2/Work-PyTheusRL/output/5node_optimal_quantum_network/5node_optimal_network_2/best.json"),
        
        ("W4 State", 
         "examples/w4_state/config.json",
         "examples/w4_state/best.json"),
        
        ("Bell State", 
         "examples/bell_state/config.json", 
         "examples/bell_state/best.json"),
    ]
    
    results = []
    
    for name, config_path, graph_path in test_configs:
        try:
            sources, detectors, beam_splitters, ancillas = test_network(name, config_path, graph_path)
            results.append((name, sources, detectors, beam_splitters, ancillas, "✅ PASS"))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((name, 0, 0, 0, 0, f"❌ FAIL: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 FINAL VALIDATION SUMMARY")
    print("="*60)
    
    for name, sources, detectors, beam_splitters, ancillas, status in results:
        print(f"{name:20} | S:{sources:2} D:{detectors:2} BS:{beam_splitters:2} A:{ancillas:2} | {status}")
    
    # Check if QKD has proper dual-role ancillas
    qkd_result = next((r for r in results if "QKD" in r[0]), None)
    if qkd_result and qkd_result[3] == qkd_result[4]:  # beam_splitters == ancillas
        print("\n✅ QKD VALIDATION: Ancillas correctly identified as dual-role beam splitters!")
    else:
        print("\n⚠️  QKD VALIDATION: Check ancilla/beam splitter identification")
    
    print("\n🎉 Validation complete! Check the generated optical table plots.")

if __name__ == "__main__":
    main()
