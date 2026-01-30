## How to Run the Tests:

1. Run all tests:

'''python
# From the repository root
python -m pytest tests/test_robust_layernorm.py -v
'''
2. Run specific test:

'''python
python -m pytest tests/test_robust_layernorm.py::test_forward_pass_2d -v
'''

3. Run with coverage:

'''python
python -m pytest tests/test_robust_layernorm.py --cov=robust_layernorm
'''

4. Direct execution:

'''python
python tests/test_robust_layernorm.py
'''

Test Categories Covered:

Initialization tests - Correct parameter creation

Forward pass tests - Shape preservation, numerical stability

Entropy estimation - Noise detection functionality

Gradient tests - Backpropagation through all parameters

Integration tests - Works with optimizers, models

Device tests - CPU/GPU compatibility

    Reproducibility tests - Deterministic behavior

    State management tests - Save/load functionality
