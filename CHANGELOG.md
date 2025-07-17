# Changelog

All notable changes to the PyTheus Modular Quantum Network Interpreter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-17

### Added
- Initial release of the PyTheus Modular Quantum Network Interpreter
- Modular architecture supporting tested network classes:
  - Single-photon source networks (W4 states)
  - Heralded Bell state preparations
  - Multi-dimensional GHZ states
  - Multi-party QKD architectures
- Dual visualization system (native graphs and optical tables)
- Multi-priority functional role identification
- Comprehensive network analysis and validation
- Journal article and validation examples
- Full API documentation and usage examples

### Features
- `ModularQuantumNetworkInterpreter` class for network analysis
- Automated functional role identification (sources, detectors, beam splitters, ancillas)
- Optical table generation with PyTheus-compatible styling
- Native graph visualization with proper edge coloring and thickness
- Comprehensive analysis reports with network metrics
- Batch processing capabilities for multiple networks
- Support for both file-based and in-memory data structures

### Validation
- Tested on 4 major PyTheus network classes
- Cross-network validation ensuring consistent results
- Self-consistency checks between mathematical and physical representations
- Performance validation across different network sizes

### Documentation
- Complete API reference
- Usage examples and demonstrations
- Journal article with technical details
- Installation and setup instructions
- Contributing guidelines

### Dependencies
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- networkx >= 2.6.0
- scipy >= 1.7.0 (optional)
- sympy >= 1.8.0 (optional)
