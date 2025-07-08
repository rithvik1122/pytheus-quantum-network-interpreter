# Contributing to PyTheus Quantum Network Interpreter

We welcome contributions to the PyTheus Quantum Network Interpreter! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug Reports**: Found a bug? Please report it!
- **Feature Requests**: Have an idea for a new feature? We'd love to hear it!
- **Code Contributions**: Help improve the codebase
- **Documentation**: Help improve documentation and examples
- **Testing**: Add test cases or test the software with new networks

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of quantum networks and Python

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/pytheus-quantum-network-interpreter.git
   cd pytheus-quantum-network-interpreter
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Example Function:

```python
def analyze_network_topology(graph_data: dict, threshold: float = 0.5) -> dict:
    """
    Analyze the topology of a quantum network graph.
    
    Args:
        graph_data: Dictionary containing graph edge information
        threshold: Minimum coupling strength to consider (default: 0.5)
    
    Returns:
        Dictionary containing topology analysis results
        
    Raises:
        ValueError: If graph_data is empty or invalid
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
python -m pytest tests/ -v
```

### Adding Tests

- Add test files in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Test both normal operation and edge cases

### Example Test:

```python
def test_load_config_from_file():
    """Test loading configuration from a valid JSON file."""
    interpreter = GeneralQuantumNetworkInterpreter()
    result = interpreter.load_config("examples/5node_qkd_network/config.json")
    assert result is True
    assert interpreter.config is not None
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Update README.md if adding new features

### Examples

- Add example scripts to the `examples/` directory
- Include both simple and advanced usage examples
- Provide sample data when possible
- Document expected outputs

## Submitting Changes

### Pull Request Process

1. **Ensure tests pass**: Run the test suite before submitting
2. **Update documentation**: Update relevant documentation
3. **Create descriptive commits**: Use clear, descriptive commit messages
4. **Submit pull request**: Create a PR with a clear description

### Commit Message Format

```
type(scope): brief description

Longer description if needed.

Fixes #issue_number (if applicable)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes

### Example Commits:

```
feat(interpreter): add support for W-state networks

Add functionality to analyze W-state quantum networks with 
symmetric multipartite entanglement structure.

Fixes #42
```

```
fix(visualization): correct edge thickness scaling

Edge thickness now properly scales with coupling strength
magnitude instead of absolute value.

Fixes #38
```

## Issue Guidelines

### Bug Reports

Please include:
- Python version and operating system
- Full error message and stack trace
- Minimal code example that reproduces the issue
- Expected vs. actual behavior
- Steps to reproduce

### Feature Requests

Please include:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant references or examples

## Code Review Process

1. **Automated checks**: All PRs run automated tests
2. **Maintainer review**: Core maintainers will review your code
3. **Discussion**: Address any feedback or questions
4. **Approval**: Once approved, your PR will be merged

## Recognition

Contributors will be:
- Listed in the AUTHORS file
- Mentioned in release notes for significant contributions
- Credited in relevant documentation

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: research.team@quantum-lab.org for private inquiries

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Guidelines

- **Be respectful**: Treat all community members with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together towards common goals
- **Be patient**: Help newcomers learn and grow

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

Thank you for contributing to the PyTheus Quantum Network Interpreter! Your contributions help advance quantum network research and development.
