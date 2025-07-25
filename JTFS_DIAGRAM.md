# JTFS Transform Computational Structure

This diagram illustrates the Joint Time-Frequency Scattering Transform (JTFS) used in the Perceptual-Neural-Physical (PNP) pipeline for audio analysis.

## Mermaid Diagram

```mermaid
graph TD
    A[Audio Signal x<br/>65,536 samples] --> B["S₀x = x * φ<br/>Lowpass Filter"]

    A --> C1["First-Order Scattering<br/>abs(x * ψλ₁)"]
    A --> C2["First-Order Scattering<br/>abs(x * ψλ₂)"]
    A --> C3[...]
    A --> C12["First-Order Scattering<br/>abs(x * ψλ₁₂)"]

    C1 --> D1["Sx(t,λ₁) = abs(x * ψλ₁) * φ"]
    C2 --> D2["Sx(t,λ₂) = abs(x * ψλ₂) * φ"]
    C3 --> D3[...]
    C12 --> D12["Sx(t,λ₁₂) = abs(x * ψλ₁₂) * φ"]

    C1 --> E11["Second-Order<br/>abs(abs(x * ψλ₁) * ψμ₁)"]
    C1 --> E12["Second-Order<br/>abs(abs(x * ψλ₁) * ψμ₂)"]
    C1 --> E13[...]

    C2 --> E21["Second-Order<br/>abs(abs(x * ψλ₂) * ψμ₁)"]
    C2 --> E22[...]

    C12 --> E121[...]

    E11 --> F11["Sx(t,λ₁,μ₁)<br/>= abs(abs(x * ψλ₁) * ψμ₁) * φ"]
    E12 --> F12["Sx(t,λ₁,μ₂)<br/>= abs(abs(x * ψλ₁) * ψμ₂) * φ"]
    E13 --> F13[...]
    E21 --> F21[...]
    E22 --> F22[...]
    E121 --> F121[...]

    B --> G[Path 1: S₀]
    D1 --> H1["Path 2-138:<br/>First-order paths<br/>137 total"]
    D2 --> H1
    D3 --> H1
    D12 --> H1

    F11 --> I["Path 139-289:<br/>Second-order joint<br/>time-frequency paths<br/>151 total"]
    F12 --> I
    F13 --> I
    F21 --> I
    F22 --> I
    F121 --> I

    G --> J["Final Coefficients<br/>289 scattering paths"]
    H1 --> J
    I --> J

    J --> K["Log Transform<br/>log₁₊₍₁₀₀₀·Sx₎"]
    K --> L["JTFS Output<br/>20,762 coefficients"]

    style A fill:#e1f5fe,color:#000000
    style L fill:#f3e5f5,color:#000000
    style G fill:#fff3e0,color:#000000
    style H1 fill:#fff3e0,color:#000000
    style I fill:#fff3e0,color:#000000
    style J fill:#e8f5e8,color:#000000

    classDef firstOrder fill:#ffecb3,stroke:#ff8f00,color:#000000
    classDef secondOrder fill:#fce4ec,stroke:#c2185b,color:#000000
    classDef output fill:#e8f5e8,stroke:#388e3c,color:#000000

    class C1,C2,C3,C12,D1,D2,D3,D12 firstOrder
    class E11,E12,E13,E21,E22,E121,F11,F12,F13,F21,F22,F121 secondOrder
    class J,K,L output
```

## Key Parameters

**JTFS Configuration (from `src/pnp_synth/utils.py`)**:
- **J=13**: 13 temporal scales (j=0 to j=13)
- **Q=(12,1)**: 12 filters per octave for first-order, 1 for second-order
- **Q_fr=1**: 1 filter per octave in frequency dimension
- **F=2**: Local frequency averaging factor
- **Shape=(2¹⁶,)**: 65,536 sample input (FTM drum synthesis)

## Computational Structure

### Three Main Components

1. **Lowpass Path (1 path)**:
   - Direct filtering: `S₀x = x * φ`
   - Captures overall signal energy

2. **First-Order Scattering (137 paths)**:
   - Computation: `Sx(t,λ) = |x * ψλ| * φ`
   - Captures temporal modulations at different scales
   - 12 filters per octave × ~12 octaves ≈ 137 paths

3. **Second-Order Scattering (151 paths)**:
   - Computation: `Sx(t,λ,μ) = ||x * ψλ| * ψμ| * φ`
   - Captures joint time-frequency modulations
   - Cross-interactions between different temporal scales

### Path Distribution Across Scales

From computational analysis (289 total paths):
- **j=0**: 14 paths (finest temporal scale)
- **j=1-7**: 21-25 paths each
- **j=8,9**: 27 paths each (peak)
- **j=10-11**: 19-24 paths
- **j=12**: 2 paths
- **j=13**: 9 paths (coarsest temporal scale)

### Final Processing

- **Log compression**: `log₁₊₍₁₀₀₀·Sx₎` for numerical stability
- **Global averaging**: Temporal averaging for translation invariance
- **Output**: 20,762 perceptual coefficients

## Computational Complexity

This multi-scale structure explains the **500× computational overhead** for Jacobian computation:

- **Forward pass**: ~10ms (direct wavelet transforms)
- **Jacobian pass**: ~5000ms (automatic differentiation through all 289 paths)
- **Memory usage**: 100MB+ during differentiation vs 300KB forward

The dense dependency graph where every output coefficient depends on all input parameters creates exponential computational complexity for gradient computation.

## Perceptual Coverage

- **Frequency range**: 2.7 Hz to 11,025 Hz (12 octaves)
- **Temporal scales**: 1ms to 1000ms modulations
- **Musical relevance**: Spans sub-bass to high harmonics
- **Perceptual modeling**: Captures complex timbre characteristics for audio synthesis parameter estimation

## [Compute Time Analysis]( ./taslp23/README_COMPUTE_TIME.md)
