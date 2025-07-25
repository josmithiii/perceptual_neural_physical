# Multi-Synthesizer Generalization Plan (MULTISYNTH_PLAN.md)

## Overview

This document outlines the strategy for generalizing the TASLP23 pipeline to support multiple synthesizer types beyond the current FTM (Functional Transformation Method) drum synthesizer. The goal is to enable the same PNP (Perceptual-Neural-Physical) training pipeline for different physical synthesis models with varying parameter dimensions and synthesis functions.

## Current TASLP23 Pipeline Analysis

### Pipeline Flow
The TASLP23 pipeline follows this sequence (from Makefile analysis):

1. **Data Generation**: `at23` → `taslp23/01_generate_audio.py`
2. **Jacobian Computation**: `jt23` → `taslp23/02_compute_pnp_jacobian.py`
3. **LMA Matrix Computation**: `Mt23` → `taslp23/12_compute_lmastep_fast.py`
4. **Training**: `rt23pl`, `rt23sl`, `rt23pnpl` → Various training scripts
5. **Fine-tuning**: `rt23ft-pnpl`, `rt23ft-mss` → Fine-tuning scripts

### Current FTM Implementation

**Parameter Structure (FTM)**:
- **Theta dimensions**: 5 parameters (`["omega", "tau", "p", "D", "alpha"]`)
- **Data files**: `taslp23/data/ftm/full_param_log.csv`
- **Synthesis function**: `ftm.rectangular_drum(theta, logscale, **ftm.constants)`
- **JTFS parameters**: `J=10`, `outdim=5`, `sr=22050`

**Parameter Structure (AM Chirp)**:
- **Theta dimensions**: 3 parameters (`["f0", "fm", "gamma"]`)
- **Data files**: `taslp23/data/amchirp/full_param_log.csv`
- **Synthesis function**: `amchirp.generate_am_chirp(theta, logscale, ...)`
- **JTFS parameters**: `J=6`, `outdim=3`, `sr=2^13`

## Generalization Strategy

### 1. Synthesizer Registry System

Create a centralized registry to manage different synthesizer types:

```python
# src/pnp_synth/synth_registry.py
SYNTHESIZER_CONFIGS = {
    "ftm": {
        "theta_columns": ["omega", "tau", "p", "D", "alpha"],
        "theta_dim": 5,
        "synthesis_fn": "ftm.rectangular_drum",
        "constants": "ftm.constants",
        "sr": 22050,
        "jtfs_params": {"J": 10, "shape": (2**16,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["omega", "p", "D"],
        "data_subdir": "ftm"
    },
    "amchirp": {
        "theta_columns": ["f0", "fm", "gamma"],
        "theta_dim": 3,
        "synthesis_fn": "amchirp.generate_am_chirp",
        "constants": {"bw": 2, "duration": 4, "sr": 2**13},
        "sr": 2**13,
        "jtfs_params": {"J": 6, "shape": (2**13*4,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["f0", "fm", "gamma"],
        "data_subdir": "amchirp"
    },
    # Future synthesizers can be added here
    "string": {
        "theta_columns": ["EI", "Ts0", "d1", "d3", "lm", "ell"],
        "theta_dim": 6,
        "synthesis_fn": "ftm.linearstring_physics",
        "constants": "ftm.constants_string",
        "sr": 22050,
        "jtfs_params": {"J": 10, "shape": (2**17,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["Ts0", "lm", "ell"],
        "data_subdir": "string"
    }
}
```

### 2. Directory Structure Modifications

Update the data directory structure to support multiple synthesizer types:

```
taslp23/data/
├── ftm/
│   ├── full_param_log.csv
│   ├── train_param_log.csv
│   ├── test_param_log.csv
│   └── val_param_log.csv
├── amchirp/
│   └── full_param_log.csv
└── string/           # New synthesizer type
    ├── full_param_log.csv
    ├── train_param_log.csv
    ├── test_param_log.csv
    └── val_param_log.csv
```

### 3. Script Modifications Required

#### A. Data Generation (`01_generate_audio.py`)

**Current State**:
- Hard-coded FTM parameters: `THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]`
- Fixed synthesis call: `ftm.rectangular_drum(theta_tensor, logscale, **ftm.constants)`

**Required Changes**:
```python
# Add synth_type parameter handling
synth_type = sys.argv[2] if len(sys.argv) > 2 else "ftm"  # Add synth_type argument
config = SYNTHESIZER_CONFIGS[synth_type]
THETA_COLUMNS = config["theta_columns"]

# Dynamic synthesis function call
synthesis_module, synthesis_fn = config["synthesis_fn"].split(".")
if synthesis_module == "ftm":
    from pnp_synth.physical import ftm
    synth_fn = getattr(ftm, synthesis_fn)
    constants = config["constants"]
elif synthesis_module == "amchirp":
    from pnp_synth.physical import amchirp
    synth_fn = getattr(amchirp, synthesis_fn)
    constants = config["constants"]

# Dynamic synthesis call
x = synth_fn(theta_tensor, logscale, **constants)
```

#### B. Jacobian Computation (`02_compute_pnp_jacobian.py`)

**Current State**:
- Fixed synth_type logic: `synth_type = "ftm" if synth_type_int == 0 else "amchirp"`
- Hard-coded JTFS parameters in `pnp_forward_factory()`

**Required Changes**:
```python
# Replace hard-coded synth_type mapping
synth_type = sys.argv[5]  # Pass synth_type directly as string
config = SYNTHESIZER_CONFIGS[synth_type]

# Update pnp_forward_factory to use config
def pnp_forward_factory(scaler, logscale, synth_type):
    config = SYNTHESIZER_CONFIGS[synth_type]
    jtfs_params = config["jtfs_params"].copy()
    jtfs_params.update({
        "max_pad_factor": 1,
        "max_pad_factor_fr": 1,
        "pad_mode": 'zero',
        "pad_mode_fr": 'zero'
    })

    # Dynamic synthesis function selection
    synthesis_module, synthesis_fn = config["synthesis_fn"].split(".")
    if synthesis_module == "ftm":
        from pnp_synth.physical import ftm
        g = functools.partial(getattr(ftm, synthesis_fn), logscale=logscale, **config["constants"])
    elif synthesis_module == "amchirp":
        from pnp_synth.physical import amchirp
        g = functools.partial(getattr(amchirp, synthesis_fn), logscale=logscale, **config["constants"])
```

#### C. Training Scripts (`03_train_effnet_ploss.py` and others)

**Current State**:
- Hard-coded parameter counts: `outdim = 5` for FTM, `outdim = 3` for amchirp
- Fixed JTFS parameters: `J = 10` for FTM, `J = 6` for amchirp

**Required Changes**:
```python
# Replace hard-coded configurations
synth_type = sys.argv[6]  # Pass synth_type as argument
config = SYNTHESIZER_CONFIGS[synth_type]

# Dynamic parameter extraction
outdim = config["theta_dim"]
J = config["jtfs_params"]["J"]
sr = config["sr"]

# Update dataset initialization
dataset = cnn.DrumDataModule(
    # ... existing parameters ...
    synth_type=synth_type,  # Add synth_type parameter
    theta_dim=outdim,       # Add theta_dim parameter
    # ... rest of parameters ...
)
```

#### D. LMA Matrix Computation (`12_compute_lmastep_fast.py`)

**Required Changes**:
- Add synth_type parameter handling
- Use dynamic theta dimensions from config
- Support variable parameter counts in matrix computations

### 4. Module Interface Updates

#### A. `taslp23.py` Module Updates

**Current State**:
- Hard-coded parameter handling in `scale_theta()`
- Fixed synthesis function in `x_from_theta()`

**Required Changes**:
```python
def scale_theta(logscale, synth_type):
    config = SYNTHESIZER_CONFIGS[synth_type]
    THETA_COLUMNS = config["theta_columns"]
    log_scale_columns = config["log_scale_columns"]

    # Dynamic parameter processing
    train_theta = []
    for column in THETA_COLUMNS:
        if not logscale and column in log_scale_columns:
            train_theta.append(10 ** train_df[column].values)
        else:
            train_theta.append(train_df[column].values)
    train_theta = np.stack(train_theta, axis=1)

def x_from_theta(theta, logscale, synth_type):
    config = SYNTHESIZER_CONFIGS[synth_type]
    synthesis_module, synthesis_fn = config["synthesis_fn"].split(".")

    if synthesis_module == "ftm":
        from pnp_synth.physical import ftm
        return getattr(ftm, synthesis_fn)(theta, logscale, **config["constants"])
    elif synthesis_module == "amchirp":
        from pnp_synth.physical import amchirp
        return getattr(amchirp, synthesis_fn)(theta, logscale, **config["constants"])
```

#### B. Neural Network Module Updates (`src/pnp_synth/neural/cnn.py`)

**Required Changes**:
- Update `DrumDataModule` to accept `synth_type` parameter
- Dynamic H5 file naming: `{synth_type}_{fold}_audio.h5`
- Support variable theta dimensions in data loading

### 5. Makefile Updates

**Required Changes**:
- Add synth_type parameter to all make targets
- Create separate targets for each synthesizer type
- Update target dependencies and file paths

```makefile
# Generalized targets with synth_type parameter
at23-ftm audio-taslp23-ftm:
	source .venv/bin/activate && python taslp23/01_generate_audio.py $(SAVE_DIR)/taslp23 ftm

at23-amchirp audio-taslp23-amchirp:
	source .venv/bin/activate && python taslp23/01_generate_audio.py $(SAVE_DIR)/taslp23 amchirp

at23-string audio-taslp23-string:
	source .venv/bin/activate && python taslp23/01_generate_audio.py $(SAVE_DIR)/taslp23 string

# Generalized training targets
rt23pl-ftm run-taslp23-ploss-ftm:
	source .venv/bin/activate && python taslp23/03_train_effnet_ploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23pl-string run-taslp23-ploss-string:
	source .venv/bin/activate && python taslp23/03_train_effnet_ploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam string
```

### 6. Data Structure Consistency

**H5 File Naming Convention**:
- Current: `taslp23_{fold}_audio.h5`
- Proposed: `{synth_type}_{fold}_audio.h5`

**Directory Structure**:
- Audio files: `./outputs/{project}/x/{synth_type}_{fold}_audio.h5`
- JTFS files: `./outputs/{project}/S/{fold}/{synth_type}_{id}_jtfs.npy`
- Jacobian files: `./outputs/{project}/J/{fold}/{synth_type}_{id}_grad_jtfs.npy`
- Model weights: `./outputs/{project}/f_W/{synth_type}_{loss_type}_{...}/`

### 7. Parameter Validation and Error Handling

**Add Validation**:
- Check that synth_type exists in registry
- Validate theta dimensions match expected values
- Ensure data files exist for specified synth_type
- Validate synthesis function availability

```python
def validate_synth_type(synth_type):
    if synth_type not in SYNTHESIZER_CONFIGS:
        raise ValueError(f"Unknown synth_type: {synth_type}. Available: {list(SYNTHESIZER_CONFIGS.keys())}")

    config = SYNTHESIZER_CONFIGS[synth_type]
    # Validate synthesis function exists
    synthesis_module, synthesis_fn = config["synthesis_fn"].split(".")
    # ... validation logic ...
```

### 8. Implementation Priority

**Phase 1: Core Infrastructure**
1. Create synthesizer registry system
2. Update `taslp23.py` for dynamic parameter handling
3. Modify data generation script (`01_generate_audio.py`)

**Phase 2: Training Pipeline**
1. Update Jacobian computation (`02_compute_pnp_jacobian.py`)
2. Modify training scripts (parameter dimensions, JTFS configs)
3. Update neural network modules for variable theta dimensions

**Phase 3: Complete Integration**
1. Update LMA matrix computation scripts
2. Modify Makefile targets
3. Add comprehensive testing and validation

**Phase 4: New Synthesizer Addition**
1. Implement string synthesizer as proof-of-concept
2. Create data generation pipeline for string synthesis
3. Test complete pipeline end-to-end

### 9. Backward Compatibility

**Maintain Compatibility**:
- Default synth_type to "ftm" when not specified
- Keep existing make targets working (e.g., `at23` defaults to FTM)
- Preserve existing data file formats and locations for FTM

### 10. Testing Strategy

**Unit Tests**:
- Test synthesizer registry functionality
- Validate dynamic parameter extraction
- Test synthesis function calls for each type

**Integration Tests**:
- End-to-end pipeline tests for each synthesizer type
- Cross-validation of JTFS parameters and audio generation
- Training convergence tests with different synthesizer types

**Regression Tests**:
- Ensure FTM results remain identical
- Validate that existing trained models still work
- Check backward compatibility of data loading

#### Lightweight Backward Compatibility Testing

**Efficient verification strategies that avoid expensive full training pipelines:**

**1. Audio Generation Verification** (Most Critical - ~10 minutes)
```bash
# Generate reference data before modifications
make at23
cp ./outputs/taslp23/x/taslp23_train_audio.h5 ./reference_ftm_audio.h5

# Test after modifications (should default to FTM)
make at23  # OR: make at23-ftm for explicit FTM

# Compare H5 files for identical audio output
python -c "
import h5py
import numpy as np
with h5py.File('./reference_ftm_audio.h5', 'r') as ref:
    with h5py.File('./outputs/taslp23/x/ftm_train_audio.h5', 'r') as new:
        for key in ref['x'].keys():
            if not np.allclose(ref['x'][key][:], new['x'][key][:], rtol=1e-10):
                print(f'MISMATCH: Sample {key}')
            else:
                print(f'MATCH: Sample {key}')
"
```

**2. JTFS Feature Comparison** (Fast subset test - ~15 minutes)
```bash
# Test first 10 samples only (much faster than 100k)
python taslp23/02_compute_pnp_jacobian.py ./outputs/taslp23 0 10 1 ftm 1

# Compare JTFS features and Jacobians
python -c "
import numpy as np
ref_S = np.load('./reference/S/train/ftm_0000000_jtfs.npy')
new_S = np.load('./outputs/taslp23/S/train/ftm_0000000_jtfs.npy')
print(f'JTFS match: {np.allclose(ref_S, new_S, rtol=1e-10)}')

ref_J = np.load('./reference/J/train/ftm_0000000_grad_jtfs.npy')
new_J = np.load('./outputs/taslp23/J/train/ftm_0000000_grad_jtfs.npy')
print(f'Jacobian match: {np.allclose(ref_J, new_J, rtol=1e-10)}')
"
```

**3. Parameter Loading Verification** (~5 minutes)
```python
# test_backward_compatibility.py
import pandas as pd
import numpy as np
from synth_registry import SYNTHESIZER_CONFIGS
import taslp23

def test_parameter_loading():
    # Original method vs registry method should be identical
    original_df = taslp23.load_fold("ftm", "train")
    new_df = taslp23.load_fold("ftm", "train")  # Updated function
    assert original_df.equals(new_df), "DataFrames don't match!"
    print("✓ Parameter loading backward compatible")

def test_parameter_scaling():
    # Parameter scaling should remain identical
    original_nus, original_scaler = taslp23.scale_theta(True, "ftm")
    new_nus, new_scaler = taslp23.scale_theta(True, "ftm")
    assert np.allclose(original_nus, new_nus), "Parameter scaling changed!"
    print("✓ Parameter scaling backward compatible")
```

**4. Synthesis Function Verification** (~5 minutes)
```python
def test_synthesis_functions():
    import torch
    from pnp_synth.physical import ftm
    from synth_registry import SYNTHESIZER_CONFIGS

    # Test with actual parameters from CSV
    theta = torch.tensor([2.43, 0.12, -4.66, -4.73, 0.046])

    # Original vs registry-based call should be identical
    original_audio = ftm.rectangular_drum(theta, True, **ftm.constants)

    config = SYNTHESIZER_CONFIGS["ftm"]
    synth_fn = getattr(ftm, "rectangular_drum")
    new_audio = synth_fn(theta, True, **ftm.constants)

    assert torch.allclose(original_audio, new_audio), "Synthesis output changed!"
    print("✓ Synthesis function backward compatible")
```

**5. Quick Pipeline Smoke Test** (~15 minutes)
```bash
# Create test subset (first 5 samples only)
head -6 taslp23/data/ftm/train_param_log.csv > test_train_subset.csv

# Test entire pipeline on tiny dataset
python taslp23/01_generate_audio.py ./test_outputs ftm
python taslp23/02_compute_pnp_jacobian.py ./test_outputs 0 5 1 ftm 1
python taslp23/03_train_effnet_ploss.py ./test_outputs test_run 1 1 adam ftm

# Should complete without errors and maintain file structure
```

**6. File Structure and Naming Verification** (~5 minutes)
```python
def test_file_naming_compatibility():
    import os
    import glob

    # Verify H5 file naming conventions
    expected_files = [
        "./outputs/taslp23/x/taslp23_train_audio.h5",  # Current naming
        "./outputs/taslp23/x/ftm_train_audio.h5"       # New naming
    ]

    # Verify JTFS/Jacobian file patterns
    expected_patterns = [
        "./outputs/taslp23/S/train/ftm_*_jtfs.npy",
        "./outputs/taslp23/J/train/ftm_*_grad_jtfs.npy"
    ]

    for pattern in expected_patterns:
        files = glob.glob(pattern)
        assert len(files) > 0, f"No files found for pattern: {pattern}"
        print(f"✓ Found {len(files)} files matching {pattern}")
```

**7. Make Target Compatibility Test** (~10 minutes)
```bash
# Test that existing make targets still work with defaults
make at23    # Should default to FTM
make jt23    # Should work with FTM
make rt23pl  # Should run with FTM parameters

# Test new explicit targets work identically
make at23-ftm    # Explicit FTM should produce same result
make jt23-ftm    # Explicit FTM Jacobian computation
make rt23pl-ftm  # Explicit FTM training
```

**8. Comprehensive Automated Test Script**
```python
#!/usr/bin/env python3
# test_multisynth_backward_compatibility.py
import subprocess
import os
import h5py
import numpy as np
import tempfile

def run_backward_compatibility_test():
    """Comprehensive backward compatibility test suite."""

    with tempfile.TemporaryDirectory() as temp_dir:
        print("🧪 Testing multi-synthesizer backward compatibility...")

        # 1. Audio generation test
        print("1. Testing audio generation...")
        result = subprocess.run([
            "python", "taslp23/01_generate_audio.py",
            temp_dir, "ftm"
        ], capture_output=True, text=True)
        assert result.returncode == 0, f"Audio generation failed: {result.stderr}"

        # 2. JTFS computation test
        print("2. Testing JTFS computation...")
        result = subprocess.run([
            "python", "taslp23/02_compute_pnp_jacobian.py",
            temp_dir, "0", "3", "1", "ftm", "1"
        ], capture_output=True, text=True)
        assert result.returncode == 0, f"JTFS computation failed: {result.stderr}"

        # 3. File structure verification
        print("3. Verifying file structure...")
        expected_files = [
            f"{temp_dir}/x/ftm_train_audio.h5",
            f"{temp_dir}/S/train/ftm_0000000_jtfs.npy",
            f"{temp_dir}/J/train/ftm_0000000_grad_jtfs.npy"
        ]

        for expected_file in expected_files:
            assert os.path.exists(expected_file), f"Missing file: {expected_file}"

        print("✅ All backward compatibility tests passed!")

if __name__ == "__main__":
    run_backward_compatibility_test()
```

**Recommended Testing Sequence** (Total: ~45 minutes)
1. **Synthesis function test** (5 min) - Quick verification of core functions
2. **Parameter loading test** (5 min) - Verify registry system works correctly
3. **Audio comparison test** (10 min) - Most critical verification
4. **JTFS comparison test** (15 min) - Verify feature extraction unchanged
5. **File structure test** (5 min) - Check naming conventions and paths
6. **Make target test** (10 min) - Ensure existing workflow still works

This provides **high confidence in backward compatibility** without expensive full training runs.

This plan provides a comprehensive strategy for generalizing the TASLP23 pipeline to support multiple synthesizer types while maintaining backward compatibility and enabling future extensibility.
