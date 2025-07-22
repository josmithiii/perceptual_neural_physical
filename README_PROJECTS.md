# Perceptual Neural Physical (PNP) Sound Matching - Project Overview (by Claude)

This repository contains a comprehensive research codebase for Perceptual-Neural-Physical (PNP) sound matching, focusing on neural parameter estimation for physics-based audio synthesis with applications to musical timbre modeling and drum synthesis.

## Main Project (Root Directory)

### Core Research
**Perceptual Neural Physical Sound Matching** by Han Han, Vincent Lostanlen, and Mathieu Lagrange

- **ICASSP Paper**: https://arxiv.org/abs/2301.02886
- **TASLP Submission**: https://arxiv.org/abs/2311.14213
- **Audio Examples**: https://lylyhan.github.io/perceptual_neural_physical/

### Key Innovation
The project introduces the **PNP Loss Function** that bridges:
1. **Physical** - Differentiable synthesis models (FTM, AMChirp)
2. **Neural** - EfficientNet-based parameter estimation
3. **Perceptual** - Joint Time-Frequency Scattering (JTFS) features

The core computation: `S = (Φ ∘ g)(θ)` where Φ is JTFS, g is synthesizer, θ are parameters.
PNP loss uses Riemannian metric weighting: `M = J^T * J` from synthesis Jacobians.

### Installation (JOS Fork of https://github.com/lylyhan/perceptual_neural_physical.git)
```bash
git clone --recurse-submodules https://github.com/josmithiii/perceptual_neural_physical.git
cd perceptual_neural_physical
sh setup.sh
```

## Conference-Based Experiment Directories

### ICASSP23 - Foundational PNP Research & Comprehensive Loss Exploration
**Focus**: Foundational ICASSP23 research establishing core PNP methodology with comprehensive exploration of loss functions, optimization strategies, and parameter scaling approaches for FTM drum synthesis

**Pipeline Structure**:
- `01_generate_audio.py` - FTM drum synthesis (5 parameters: ω, τ, p, D, α) with CPU-based synthesis
- `02_compute_pnp_jacobian.py` - JTFS Jacobians using forward-mode automatic differentiation
- `03*_train_effnet_ploss.py` - P-Loss training variants with cross-platform support (CUDA/MPS/CPU)
- `04*_train_effnet_specloss.py` - Multi-Scale Spectral Loss implementations
- `05_compute_M.py` - Riemannian metric computation from synthesis Jacobians
- `06-13_train_effnet_pnploss*.py` - Extensive PNP loss variants with Levenberg-Marquardt adaptations
- `13_compute_lmastep.py` - Advanced LMA step computation with modal analysis
- `14_merge_h5file.py` - HDF5 data consolidation for distributed processing

**Key Innovations**:
- **Adaptive LMA Scheduling**: Multiple strategies (adaptive, scheduled, constant) with accelerator/brake mechanisms
- **Modal Analysis**: Eigenvalue-based parameter space analysis with mean metric weighting
- **Comprehensive Optimization**: Identity vs mean damping, diagonal vs full matrix weighting
- **Cross-Platform Training**: CUDA, MPS (Apple Silicon), CPU compatibility with optimized device selection
- **Systematic Parameter Scaling**: Log-scale transformation + MinMax normalization with extensive ablation studies

### GRETSI23 - Tonal Synthesis & Parameter Scaling Analysis
**Focus**: PNP methodology applied to tonal synthesis (AMChirp) with systematic exploration of parameter scaling effects and optimization landscape differences between tonal vs percussive signals

**Pipeline Structure**:
- `01_generate_audio.py` - AMChirp synthesis (3 parameters: f0, fm, γ) with geometric parameter sampling (~27k samples)
- `02_compute_pnp_jacobian.py` - JTFS Jacobians for AMChirp with scaling variant support
- `03*_train_effnet_ploss*.py` - P-Loss variants exploring log vs raw parameter handling
- `04_train_effnet_pnploss_*_adaptive_brake1.py` - Adaptive PNP training for tonal synthesis
- `05*_train_effnet_pnploss_novol*.py` - PNP "no volume" variants with parameter scaling exploration
- `06*_train_effnet_specloss*.py` - Spectral loss optimized for tonal signals
- `07*_train_effnet_pnploss_*_meanM*.py` - Mean metric PNP training with dataset-averaged weighting

**Key Research Contributions**:
- **Parameter Scaling Systematic Study**: Log vs raw parameters, MinMax vs direct optimization
- **Tonal vs Percussive Comparison**: 3-parameter AMChirp vs 5-parameter FTM analysis
- **Geometric Sampling**: Intelligent parameter space exploration with validation set centering
- **Mean Metric Weighting**: Dataset-averaged Riemannian metrics for improved stability
- **Clear Naming Convention**: _nolog, _raw, _log suffixes for systematic parameter handling exploration

### TASLP23 - Advanced PNP Methodology & Fine-tuning
**Focus**: Advanced PNP methodology development with fine-tuning approaches, two-stage training, and comprehensive metric analysis

**Pipeline Structure**:
- `01_generate_audio.py` - FTM drum synthesis (5 parameters: ω, τ, p, D, α)
- `02_compute_pnp_jacobian.py` - JTFS Jacobians for PNP loss
- `03-10_train_effnet_*.py` - Multi-loss training (P-Loss, MSS, PNP) with fine-tuning variants
- `12-13_compute_*.py` - Riemannian metrics and HDF5 management

**Key Features**:
- **Fine-tuning Approaches**: Two-stage training (P-Loss → PNP, P-Loss → MSS)
- **Multi-loss Comparison**: P-Loss, Spectral Loss, PNP Loss systematic evaluation
- **Parameter Scaling**: Log-scale + MinMax normalization optimization
- **HPC Deployment**: SLURM batch scripts for distributed training
- **Raw Parameter Training**: Alternative parameter handling without MinMax scaling

### ICASSP25 - Enhanced Optimization & Analysis
**Focus**: Advanced optimization techniques, gradient analysis, and cross-platform compatibility

**Pipeline Structure**:
- `00_train.py` - Unified training framework (Adam/Sophia optimizers)
- `01_generate_audio*.py` - Enhanced FTM synthesis with MPS compatibility
- `01b_compute_pnp_jacobian.py` - Forward-mode automatic differentiation
- `03-05_train_effnet_*.py` - P-Loss and PNP training implementations
- `06-08_eval_grad*.py` - Comprehensive gradient analysis tools
- `audio_comparison.py` + `generate_comparison_html.py` - Interactive evaluation

**Key Innovations**:
- **Sophia Optimizer**: Second-order optimization for neural networks
- **Cross-Platform Support**: CUDA, MPS (Apple Silicon), CPU compatibility
- **Gradient Analysis**: Optimization landscape and convergence behavior studies
- **Interactive Evaluation**: Web-based audio comparison interfaces
- **DOCE Framework**: Design of Computer Experiments for hyperparameter exploration

**Advanced Features**:
- Adaptive Levenberg-Marquardt scheduling for PNP loss
- Real-time gradient monitoring and clipping analysis
- Parameter error visualization with color-coded indicators
- Systematic experimental design with reproducible trials

### MERSENNE24 - Noise Robustness Research
**Focus**: Noise-robust neural parameter estimation for real-world drum recordings

**Research Questions**: How do different noise augmentation strategies affect parameter estimation robustness?

**Noise Strategies**:
- **Basic Noise**: Random, Gaussian, stationary mixing with real audio
- **Matched Noise**: Acoustically-aligned real audio pairing
- **Physics-Informed Noise**: PRATM (Physical Real-Audio Transient Modeling)
- **Diffusion-Generated Noise**: Advanced generative modeling (`08_train_diffuser.py`)

**Pipeline Structure**:
- `01_generate_audio*.py` - Physics/perceptual parameter synthesis (6 params)
- `03-12_train_effnet_*.py` - Systematic noise augmentation experiments
- `08-09_*diffuser.py` - 1D UNet diffusion models for noise generation

**Key Features**:
- **Dual Parameter Modes**: Physics (EI, Ts0, d1, d3, lm, ell) vs Perceptual (w1, tau, p, D, lm, ell)
- **Position Randomization**: Randomized excitation positions for synthesis diversity
- **Comprehensive Evaluation**: Clean synthetic vs noise-augmented test sets
- **Diffusion Models**: DanceDiffusion pipeline for realistic noise generation

## Supporting Libraries

### JTFS-GPU - High-Performance Time-Frequency Scattering
**Location**: `jtfs-gpu/` (git submodule from https://github.com/cyrusvahidi/jtfs-gpu)

**Research**: "Differentiable Time-Frequency Scattering on GPU" (DAFx 2022, Best Paper Award)

**Applications**:
- Unsupervised manifold learning of spectrotemporal modulations
- Hybrid JTFS + ConvNet musical instrument classification
- Texture resynthesis with differentiable reconstruction
- K-NN regression of synthesizer parameters

**Features**:
- GPU-accelerated JTFS implementation in Kymatio
- Scale-rate visualizations for spectrotemporal analysis
- Medley-solos-DB musical instrument classification
- Isomap embeddings and manifold learning tools

### STRF-Like Model - Spectrotemporal Receptive Fields
**Location**: `jtfs-gpu/strf-like-model/`

**Research**: Thoret et al. (2020) "Learning metrics on spectrotemporal modulations reveals the perception of musical instrument timbre" (Nature Human Behaviour)

**Purpose**: Python implementation of NSL Toolbox-inspired STRF representations for perceptual analysis.

## Architecture & Data Flow

### Three-Component PNP Pipeline
```
Parameters (θ) → Physical Synthesis (g) → Audio (x) → Perceptual Features (Φ) → JTFS Coefficients (S)
                     ↓ Jacobian (J)                    ↓ Neural Network
               Riemannian Metric (M = J^T*J) ← Parameter Estimates (θ̂)
```

### Key Components
1. **Physical Models**:
   - **FTM** (drums): 5-parameter drum synthesis via 4th-order PDEs (ω, τ, p, D, α)
   - **AMChirp** (tonal): 3-parameter amplitude-modulated chirp synthesis (f0, fm, γ)
   - **String** (percussion): 6-parameter physics-based string modeling (EI, Ts0, d1, d3, lm, ell)
2. **Neural Networks**: EfficientNet (B0-B7) with specialized loss functions and cross-platform optimization
3. **Perceptual Features**: JTFS coefficients with physics-informed Riemannian metric weighting

### Parameter Scaling Strategy
- **Log-scale transformation** for frequency parameters
- **MinMax normalization** to [-1, 1] for neural compatibility
- **Differentiable inverse scaling** in synthesis forward pass

### Loss Function Hierarchy
- **P-Loss**: Basic parameter MSE (baseline)
- **Spec-Loss**: Multi-scale spectral reconstruction
- **PNP-Loss**: Perceptual-Neural-Physical with Riemannian weighting
- **Adaptive variants**: Levenberg-Marquardt scheduling

## Development Workflow

### Typical Pipeline Sequence
1. **Data Generation** (`01_generate_audio.py`) - Synthetic training data
2. **Gradient Computation** (`02_compute_pnp_jacobian.py`) - PNP loss preparation
3. **Training** (`03-07_train_effnet_*.py`) - Multiple loss strategies
4. **Evaluation** (`0X_eval_*.py`) - Comprehensive analysis
5. **Analysis** (`audio_comparison.py`, gradient analysis) - Results interpretation

### Key Dependencies
- **kymatio** (JTFS-GPU fork): Time-frequency scattering
- **pytorch-lightning**: Training framework
- **functorch**: Auto-differentiation for Jacobians
- **auraloss**: Perceptual audio loss functions
- **librosa, soundfile**: Audio processing
- **h5py**: Efficient data storage

### Data Organization
Each experiment contains:
- `data/`: Parameter logs (CSV) and HDF5 audio/gradient storage
- `sbatch/`: SLURM cluster job scripts for distributed processing
- Numbered pipeline scripts following consistent patterns (01_generate → 02_compute → 03-0X_train)
- Configuration modules (`icassp23.py`, `gretsi23/setups.py`, `taslp23.py`, `icassp25.py`, `mersenne24.py`)
- Systematic naming conventions for parameter scaling variants (_nolog, _raw, _log suffixes)

This codebase represents a comprehensive framework for physics-informed neural audio synthesis, combining cutting-edge optimization techniques with principled perceptual modeling for musical timbre research.
