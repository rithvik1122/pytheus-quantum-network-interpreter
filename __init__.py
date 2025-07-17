#!/usr/bin/env python3
"""
PyTheus Modular Quantum Network Interpreter

A modular interpreter for analyzing and visualizing PyTheus-optimized quantum networks
from tested network classes.

This package provides tools for:
- Loading and parsing PyTheus quantum network configurations
- Analyzing network topology and identifying functional components
- Generating optical table layout visualizations
- Creating native PyTheus-style graph plots
- Producing comprehensive analysis reports

Author: Rithvik
License: MIT
Version: 1.0.0
"""

from .modular_interpreter import ModularQuantumNetworkInterpreter

__version__ = "1.0.0"
__author__ = "Rithvik"
__email__ = "rithvik1122@example.com"
__license__ = "MIT"

__all__ = [
    "ModularQuantumNetworkInterpreter",
]
