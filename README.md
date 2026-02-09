# DLSA - Deep Learning Statistical Arbitrage

A modular reimplementation of the methodology from **"Deep Learning Statistical Arbitrage"** by Guijarro-Ordonez, Pelger, and Zanotti.

**Author:** Oualid Missaoui

## Features

- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Type Hints**: Full type annotations throughout the codebase
- **Comprehensive Docstrings**: Detailed documentation for all functions and classes
- **Multiple Factor Models**: IPCA, PCA, and Fama-French
- **Neural Network Models**: CNN-Transformer, Fourier FFN, OU Threshold baseline
- **Rolling Window Backtesting**: Production-ready simulation framework

## Directory Structure

```
dlsa_clean/
├── dlsa/                      # Main package
│   ├── backtesting/          # Backtesting/simulation engine
│   │   ├── results.py        # Result storage and analysis
│   │   └── simulator.py      # Rolling window simulator
│   ├── config/               # Configuration management
│   │   └── settings.py       # Config dataclasses and YAML loading
│   ├── data/                 # Data loading and datasets
│   │   ├── dataset.py        # PyTorch datasets
│   │   ├── filters.py        # Data filtering utilities
│   │   └── loader.py         # Data loading functions
│   ├── factor_models/        # Factor model implementations
│   │   ├── base.py           # Abstract base class
│   │   ├── ipca.py           # IPCA implementation
│   │   ├── pca.py            # PCA implementation
│   │   └── fama_french.py    # Fama-French implementation
│   ├── metrics/              # Performance metrics
│   │   ├── performance.py    # Sharpe, Sortino, etc.
│   │   └── statistics.py     # Statistical tests
│   ├── models/               # Neural network models
│   │   ├── base.py           # Abstract base model
│   │   ├── cnn_transformer.py
│   │   ├── fourier_ffn.py
│   │   └── ou_threshold.py
│   ├── portfolio/            # Portfolio construction
│   │   ├── policies.py       # Weight policies
│   │   └── weights.py        # Weight transformation
│   ├── preprocessing/        # Feature engineering
│   │   ├── methods.py        # Cumsum, Fourier, OU preprocessing
│   │   └── ou_params.py      # OU parameter estimation
│   ├── training/             # Training engine
│   │   ├── loss.py           # Loss functions
│   │   ├── optimizer.py      # Optimizer utilities
│   │   └── trainer.py        # Training loop
│   ├── utils/                # Utilities
│   │   ├── device.py         # GPU/device management
│   │   ├── logging.py        # Logging utilities
│   │   └── tensor_ops.py     # Tensor operations
│   └── types.py              # Type definitions and dataclasses
├── configs/                   # Example configuration files
├── run_simulation.py         # Main simulation script
├── run_factor_model.py       # Factor model computation script
├── run_stats.py              # Statistics computation script
└── README.md
```

## Installation

```bash
# Install dependencies
pip install torch numpy pandas scipy scikit-learn pyyaml

# Or with requirements.txt
pip install -r requirements.txt
```

## Data

The raw data required for this project can be downloaded from Dropbox:

**[Download Data](https://www.dropbox.com/scl/fo/di9dls4pj7p9i7jkz6ibh/ANJUVB_yM-B0hGbiv_EyVSw?rlkey=2yfxdvj5niaowja7n76icyvcg&e=1&dl=0)**

After downloading, extract the data and place it in a directory accessible to the project. Update the `data_dir` and `residuals_dir` paths in your configuration accordingly.

## Quick Start

### Using Pre-computed Residuals

If you have pre-computed residuals from the original implementation:

```bash
# Point to the original data directory
python run_simulation.py \
    --data-dir ../Deep_Learning_Statistical_Arbitrage_Code_Data/data \
    --residuals-dir ../Deep_Learning_Statistical_Arbitrage_Code_Data/residuals \
    --factor-model IPCA \
    --num-factors 5 \
    --model CNNTransformer
```

### Computing Residuals

If you need to compute residuals from raw returns:

```bash
python run_factor_model.py \
    --data-dir ./data \
    --output-dir ./residuals \
    --model IPCA \
    --factors 5 \
    --characteristics ./data/characteristics.npy
```

### Using Configuration Files

Create a YAML configuration file:

```yaml
# config.yaml
data_dir: "./data"
residuals_dir: "./data/residuals"
results_dir: "./results"

factor_model: "IPCA"
num_factors: 5

model_class: "CNNTransformer"
lookback: 30
filter_numbers: [1, 8]
attention_heads: 4

num_epochs: 100
batch_size: 125
learning_rate: 0.001
objective: "sharpe"

length_training: 1000
stride: 125
```

Then run:

```bash
python run_simulation.py --config config.yaml
```

## Python API Usage

```python
from dlsa.config import Config
from dlsa.data.loader import DataLoader
from dlsa.backtesting.simulator import Simulator
from dlsa.types import PreprocessingMethod

# Load configuration
config = Config(
    data_dir="../data",
    residuals_dir="../residuals",
    factor_model="IPCA",
    num_factors=5,
    model_class="CNNTransformer",
    num_epochs=100,
)

# Load data
data_loader = DataLoader(config.data_dir, config.residuals_dir)
residuals = data_loader.get_residuals("IPCA", 5)
comp_mtx = data_loader.get_composition_matrix("IPCA", 5)

# Create and run simulator
simulator = Simulator(
    backtest_config=config.get_backtest_config(),
    training_config=config.get_training_config(),
    model_config=config.get_model_config(),
)

results = simulator.run(
    residuals=residuals,
    composition_matrix=comp_mtx,
    preprocessing=PreprocessingMethod.CUMSUM,
)

# Print results
stats = results.compute_aggregate_stats()
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
print(f"Annual Return: {stats['mean_return']:.2%}")
```

## Models

### CNNTransformer (Default)

Hybrid architecture combining:
- 1D CNN for local pattern extraction
- Transformer encoder for cross-asset attention

```python
from dlsa.models import CNNTransformer

model = CNNTransformer(
    lookback=30,
    filter_numbers=[1, 8],
    attention_heads=4,
    dropout=0.25,
)
```

### FourierFFN

Simple feedforward network on Fourier features:

```python
from dlsa.models import FourierFFN

model = FourierFFN(
    lookback=30,
    hidden_units=[30, 16, 8, 4],
    dropout=0.25,
)
```

### OUThreshold (Baseline)

Non-trainable baseline using Ornstein-Uhlenbeck mean reversion:

```python
from dlsa.models import OUThreshold

model = OUThreshold(
    lookback=30,
    signal_threshold=1.25,
    r2_threshold=0.25,
)
```

## Factor Models

### IPCA

Instrumented PCA with time-varying loadings:

```python
from dlsa.factor_models import IPCAFactorModel

model = IPCAFactorModel(num_factors=5)
model.fit(returns, characteristics)
residuals = model.compute_residuals(returns, characteristics)
```

### PCA

Standard principal component analysis:

```python
from dlsa.factor_models import PCAFactorModel

model = PCAFactorModel(num_factors=5)
model.fit(returns)
residuals = model.compute_residuals(returns)
```

### Fama-French

Traditional factor model regression:

```python
from dlsa.factor_models import FamaFrenchFactorModel

model = FamaFrenchFactorModel(num_factors=5)
model.fit(returns, factor_returns=ff_factors)
residuals = model.compute_residuals(returns, factor_returns=ff_factors)
```

## Preprocessing Methods

- **cumsum**: Rolling cumulative return windows
- **fourier**: FFT coefficients of return windows
- **ou**: Ornstein-Uhlenbeck parameters (μ, σ, R²)

## Key Differences from Original

1. **Modular Design**: Each component is independent and testable
2. **Type Safety**: Full type hints for better IDE support
3. **Documentation**: Comprehensive docstrings
4. **No Duplication**: Shared utilities instead of copy-paste
5. **Clean Interfaces**: Well-defined boundaries between modules
6. **Data Pointers**: Points to original data directory (no copying)

## Hardware Requirements

Same as original implementation:
- **Full replication**: 16 cores, 384GB RAM, 36GB VRAM
- **Partial (with residuals)**: 4 cores, 256GB RAM, 12GB VRAM

## References

- Paper: [Deep Learning Statistical Arbitrage](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862004)
- Original Code: [GitHub Repository](https://github.com/markus-pelger/Deep_Learning_Statistical_Arbitrage)

## License

MIT License - See LICENSE file for details.
