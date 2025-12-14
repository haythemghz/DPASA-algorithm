# DPASA: Dual-Population Adversarial Strategy Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

DPASA is a novel meta-optimization framework that decouples search strategy evolution from candidate solution evolution. By maintaining two co-evolving populations—one for strategies (search behaviors) and one for candidates (solutions)—and employing an adversarial **Negation Operator**, DPASA prevents premature convergence and robustly explores deceptive landscapes.

## Features
- **Dual-Population Architecture**: Simultaneous evolution of strategy parameters and candidate solutions.
- **Adversarial Negation**: Active testing of counter-strategies to escape local optima.
- **Sinusoidal Guidance encoding**: Low-dimensional parameterization of complex search trajectories.
- **Engineering Optimization**: Specialized constraint handling for real-world design problems.

## Repository Structure
```
.
├── paper/                  # LaTeX source code for the DPASA paper
├── src/                    # Source code for the algorithm
│   ├── dpasa/              # Core algorithm implementation
│   ├── benchmarks/         # CEC and Engineering benchmark problems
│   └── visualization/      # Plotting and analysis tools
├── examples/               # Example usage scripts
├── experiments/            # Verification and reproduction scripts
├── results/                # Experimental logs and data
├── tests/                  # Unit tests
└── setup.py                # Installation script
```

## Installation

```bash
git clone https://github.com/haythemghz/DPASA-algorithm.git
cd DPASA-algorithm
pip install -e .
```

## Usage

### Basic Example
```python
import numpy as np
from dpasa import DPASAOptimizer

def sphere(x):
    return np.sum(x**2)

optimizer = DPASAOptimizer(
    objective_func=sphere,
    dim=30,
    bounds=(-100, 100),
    n_strategies=50,
    n_candidates=500,
    max_iterations=100
)

result = optimizer.optimize()
print(f"Best Solution: {result['best_solution']}")
print(f"Best Fitness: {result['best_fitness']}")
```

### Running Engineering Benchmarks
```bash
python examples/run_engineering.py
```

## Citation
If you use DPASA in your research, please cite:
```bibtex
@article{ghazouani2025dpasa,
  title={Dual-Population Adversarial Strategy Adaptation (DPASA): A Meta-Optimization Framework for Global Optimization},
  author={Ghazouani, Haythem},
  journal={Swarm and Evolutionary Computation},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
