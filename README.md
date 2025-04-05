# Mamba: Efficient State Space Model with Triton-accelerated Selective Scans

Mamba is a modern state space model (SSM) featuring **input-dependent state transitions** and **hardware-aware parallel scans** using Triton. This implementation demonstrates high-performance sequence modeling through a combination of causal convolutions, selective parameterization, and GPU-optimized recurrent computations.


## Key Features
- 🚀 **Triton-accelerated selective scans** for parallelized recurrent computations
- ⏳ **Input-dependent system parameters** (Δ, Ã, B̃, C̃) via learned projections
- ⚡ **Causal depthwise convolution** for local feature extraction
- 🧮 **Structured state matrices** with complex-number initialization
- 📈 **Memory-efficient design** with O(L) memory complexity

## Usage

### Basic Model Initialization
```python
from model import Mamba, SSMConfig

config = SSMConfig(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2
)
model = Mamba(config).cuda()

# Forward pass example
x = torch.randn(8, 1024, 512).cuda()  # (batch, seq_len, dim)
output = model(x)
```

### Core Components
```python
# Causal convolution layer
x = model.conv1d(x)  # Maintains temporal causality

# Dynamic parameter generation
delta, A_mod, B_mod, C_mod = model.s_proj(x)  # Input-dependent parameters

# Discretized state space system
A_disc, B_disc = model.discretization(delta)  # Continuous-to-discrete conversion

# Triton-accelerated selective scan
y = model.selective_scan(x, delta, A_disc, B_disc, C_mod)
```

## Model Architecture
| Component               | Specification                          |
|-------------------------|----------------------------------------|
| Hidden Dimension        | 512                                   |
| State Dimension         | 16                                    |
| Convolution Kernel      | 4                                     |
| Expansion Factor        | 2                                     |
| Sequence Length         | ≤2048 (theoretically unbounded)       |

## Training Configuration
- **Parameter Initialization**:
  - Xavier normal for linear layers
  - Kaiming normal for convolutional layers
- **Dynamic Parameter Activation**:
  - Softplus for Δ (time step scale)
  - Sigmoid for A modifications
- **System Constraints**:
  - Causal convolution padding
  - Complex-number state matrix initialization

## Performance Optimizations
1. **Triton Kernel Features**:
   - Block-wise parallel processing (16 model dim × 8 state dim blocks)
   - Shared memory caching for hidden states
   - Double buffering for memory latency hiding

2. **Memory Management**:
   - In-place operations for state updates
   - Depthwise separable convolutions
   - Selective parameter recomputation

## Theoretical Complexity
| Operation                | Time        | Space       |
|--------------------------|-------------|-------------|
| Convolution              | O(L·D²)     | O(L·D)      |
| Selective Scan           | O(L·D·N)    | O(D·N)      |
| Total                    | O(L·D²)     | O(L·D)      |

Where:
- L: Sequence length
- D: Model dimension (d_model)
- N: State dimension (d_state)

## License
[Apache 2.0](LICENSE) - Open for academic and commercial use with attribution.

---

**Note**: This implementation focuses on demonstrating the core Mamba concepts. For production use, consider:
- Adding normalization layers
- Implementing hybrid precision training
- Incorporating attention mechanisms for global context
```
