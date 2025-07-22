# Perceptual Neural Physical Sound Matching - Research Timeline (according to Detective Claude)

This document provides a comprehensive timeline of the PNP
(Perceptual-Neural-Physical) sound matching research development, with
analysis of what the "Current" method might refer to in the [online sound
matching
examples](https://lylyhan.github.io/perceptual_neural_physical/).

## Research Development Timeline

### **October 2022 - ICASSP23: Foundational Research**
**Directory**: `icassp23/` **Status**: Published foundational work

- **Innovation**: Introduction of the core PNP Loss Function bridging Physical synthesis, Neural networks, and Perceptual features
- **Synthesis**: 5-parameter FTM drum synthesis (ω, τ, p, D, α) via 4th-order PDEs
- **Key Contributions**:
  - Riemannian metric weighting from synthesis Jacobians (M = J^T * J)
  - Comprehensive loss function exploration (P-Loss, Spectral Loss, PNP Loss)
  - Adaptive Levenberg-Marquardt scheduling strategies
  - Cross-platform compatibility (CUDA/MPS/CPU)
- **Optimization**: Multiple LMA variants (adaptive, scheduled, constant) with accelerator/brake mechanisms
- **Publication**: [ICASSP Paper](https://arxiv.org/abs/2301.02886)

### **2023 - GRETSI23: Tonal Synthesis Exploration**
**Directory**: `gretsi23/` **Status**: Conference proceedings

- **Focus**: PNP methodology applied to **tonal synthesis** rather than percussive
- **Synthesis**: 3-parameter AMChirp (f0, fm, γ) with geometric parameter sampling (~27k samples)
- **Key Research**:
  - Systematic parameter scaling analysis (log vs raw, MinMax vs direct)
  - Tonal vs percussive signal comparison for PNP effectiveness
  - Mean metric weighting with dataset-averaged Riemannian metrics
  - Intelligent parameter space exploration with validation set centering
- **Innovation**: Clear naming conventions (_nolog, _raw, _log) for systematic ablation studies

### **June 2023 - TASLP23: Advanced Methodology & Refinement**
**Directory**: `taslp23/` **Status**: Published journal article ⭐ **LIKELY "CURRENT" METHOD**

- **Focus**: **Advanced PNP methodology** development with sophisticated training approaches
- **Key Innovations**:
  - **Fine-tuning approaches**: Two-stage training (P-Loss → PNP, P-Loss → MSS)
  - **Comprehensive method comparison**: P-Loss, Spectral Loss, PNP Loss systematic evaluation
  - **Raw parameter training**: Alternative parameter handling without MinMax scaling
  - **HDF5 data consolidation**: Efficient distributed processing for large-scale training
- **Synthesis**: Enhanced 5-parameter FTM drum synthesis with advanced optimization
- **Publication**: [TASLP Submission](https://arxiv.org/abs/2311.14213)
- **Assessment**: **Most likely the "Current" method** demonstrated on the website

### **May 2024 - MERSENNE24: Noise Robustness Research**
**Directory**: `mersenne24/` **Status**: Ongoing research

- **Focus**: Noise-robust neural parameter estimation for real-world applications
- **Synthesis**: **6-parameter string synthesis** - most advanced physical modeling
  - **Physics mode**: EI, Ts0, d1, d3, lm, ell (bending stiffness, tension, damping)
  - **Perceptual mode**: w1, τ, p, D, lm, ell (perceptual parameter mapping)
- **Noise Strategies**:
  - Matched noise (acoustically-aligned real audio pairing)
  - Physics-informed PRATM (Physical Real-Audio Transient Modeling)
  - Diffusion-generated noise using 1D UNet models
- **Innovation**: Position randomization for synthesis diversity, dual parameter modes

### **2025 - ICASSP25: Enhanced Optimization & Analysis**
**Directory**: `icassp25/` **Status**: Current submission

- **Focus**: Advanced optimization techniques and comprehensive analysis tools
- **Key Features**:
  - **Sophia Optimizer**: Second-order optimization for neural networks
  - **Gradient analysis**: Optimization landscape and convergence behavior studies
  - **Interactive evaluation**: Web-based audio comparison interfaces
  - **DOCE Framework**: Design of Computer Experiments for systematic hyperparameter exploration
- **Analysis Tools**: Real-time gradient monitoring, parameter error visualization, systematic experimental design

## "Current" Method Analysis

### **Evidence for TASLP23 as "Current"**

Based on git commit analysis and research timeline investigation:

#### **1. Timeline Evidence**
- **Active development through 2025**: Git commits show Han (lylyhan) actively updating TASLP23-related components
- **Recent main branch commits**: "mersenne data v3", "update dataset", "ploss" (March 2025)
- **Continuous refinement**: TASLP23 represents the mature, refined version of the original ICASSP23 approach

#### **2. Methodological Sophistication**
- **Two-stage training**: Most advanced training methodology with P-Loss → PNP fine-tuning
- **Comprehensive optimization**: Multiple loss function variants systematically evaluated
- **Production-ready**: Refined parameter handling and efficient data processing pipelines

#### **3. Website Context**
The [online examples](https://lylyhan.github.io/perceptual_neural_physical/) show:
- **"Target"**: Ground truth audio
- **"Current"**: Most likely **TASLP23 fine-tuned PNP** method ⭐
- **"Ploss"**: Baseline parameter loss method
- **"PNP ICASSP"**: Original ICASSP23 PNP implementation

### **Research Progression Demonstration**

The website serves as a **research evolution showcase**:

```
ICASSP23 PNP (foundational) → TASLP23 "Current" (refined) → Future work
     ↓                              ↓
Original methodology        State-of-the-art results
```

## Synthesis Model Evolution

### **Parameter Complexity Growth**
1. **GRETSI23**: 3-parameter AMChirp (f0, fm, γ) - Tonal synthesis
2. **ICASSP23/TASLP23**: 5-parameter FTM (ω, τ, p, D, α) - Drum synthesis
3. **MERSENNE24**: 6-parameter String (EI, Ts0, d1, d3, lm, ell) - Advanced percussion

### **Physical Modeling Advancement**
- **AMChirp**: Amplitude-modulated chirp generation
- **FTM**: 4th-order partial differential equation drum modeling
- **String**: Physics-based string modeling with bending stiffness and tension control

## Development Philosophy

### **Conference-Driven Organization**
Each directory represents a **complete research contribution**:
- Self-contained experiment pipelines
- Reproducible configuration and data management
- Systematic naming conventions for ablation studies

### **Methodological Evolution**
- **ICASSP23**: "Can we bridge physical synthesis with neural parameter estimation?"
- **GRETSI23**: "How does PNP work for tonal vs percussive signals?"
- **TASLP23**: "How can we optimize and refine the PNP approach?" ⭐
- **MERSENNE24**: "How robust is PNP to real-world noise conditions?"
- **ICASSP25**: "What advanced optimization techniques can further improve PNP?"

## Conclusion

**"Current" most likely refers to the TASLP23 fine-tuned PNP method**, representing the culmination of the foundational ICASSP23 research with advanced two-stage training, comprehensive optimization, and refined parameter handling. This makes TASLP23 the **current state-of-the-art** at the time the website examples were generated, demonstrating the significant improvements achieved through methodological refinement and sophisticated training approaches.

The timeline shows a clear research trajectory from **foundational innovation** (ICASSP23) through **systematic exploration** (GRETSI23) to **methodological refinement** (TASLP23), with ongoing work on **practical robustness** (MERSENNE24) and **advanced optimization** (ICASSP25).
