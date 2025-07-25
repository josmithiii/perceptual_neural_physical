"""
Synthesizer Registry System for Multi-Synthesizer Support

This module provides a centralized registry to manage different synthesizer types
in the PNP (Perceptual-Neural-Physical) pipeline, enabling support for multiple
physical synthesis models with varying parameter dimensions and synthesis functions.
"""

SYNTHESIZER_CONFIGS = {
    "ftm": {
        "theta_columns": ["omega", "tau", "p", "D", "alpha"],
        "theta_dim": 5,
        "synthesis_fn": "ftm.rectangular_drum",
        "constants": "ftm.constants",
        "sr": 22050,
        "jtfs_params": {"J": 13, "shape": (2**16,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["omega", "p", "D"],
        "data_subdir": "ftm",
        "description": "Functional Transformation Method drum synthesizer"
    },
    "amchirp": {
        "theta_columns": ["f0", "fm", "gamma"],
        "theta_dim": 3,
        "synthesis_fn": "amchirp.generate_am_chirp",
        "constants": {"bw": 2, "duration": 4, "sr": 2**13},
        "sr": 2**13,
        "jtfs_params": {"J": 13, "shape": (2**13*4,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["f0", "fm", "gamma"],
        "data_subdir": "amchirp",
        "description": "Amplitude-modulated chirp synthesizer"
    },
    "string": {
        "theta_columns": ["EI", "Ts0", "d1", "d3", "lm", "ell"],
        "theta_dim": 6,
        "synthesis_fn": "ftm.linearstring_physics",
        "constants": "ftm.constants_string",
        "sr": 22050,
        "jtfs_params": {"J": 13, "shape": (2**17,), "Q": (12, 1), "Q_fr": 1, "F": 2},
        "log_scale_columns": ["Ts0", "lm", "ell"],
        "data_subdir": "string",
        "description": "Linear string physical model synthesizer"
    }
}


def validate_synth_type(synth_type: str) -> None:
    """
    Validate that a synthesizer type exists in the registry.

    Args:
        synth_type: String identifier for synthesizer type

    Raises:
        ValueError: If synth_type is not in registry
    """
    if synth_type not in SYNTHESIZER_CONFIGS:
        available = list(SYNTHESIZER_CONFIGS.keys())
        raise ValueError(f"Unknown synth_type: {synth_type}. Available: {available}")


def get_synth_config(synth_type: str) -> dict:
    """
    Get configuration dictionary for a synthesizer type.

    Args:
        synth_type: String identifier for synthesizer type

    Returns:
        Configuration dictionary containing synthesis parameters

    Raises:
        ValueError: If synth_type is not in registry
    """
    validate_synth_type(synth_type)
    return SYNTHESIZER_CONFIGS[synth_type].copy()


def get_synthesis_function(synth_type: str):
    """
    Get the synthesis function for a given synthesizer type.

    Args:
        synth_type: String identifier for synthesizer type

    Returns:
        Tuple of (module, function, constants) for synthesis

    Raises:
        ValueError: If synth_type is not in registry
        ImportError: If synthesis module cannot be imported
    """
    config = get_synth_config(synth_type)
    synthesis_module, synthesis_fn = config["synthesis_fn"].split(".")

    if synthesis_module == "ftm":
        from pnp_synth.physical import ftm
        synth_fn = getattr(ftm, synthesis_fn)
        constants = getattr(ftm, config["constants"].split(".")[1]) if isinstance(config["constants"], str) else config["constants"]
    elif synthesis_module == "amchirp":
        from pnp_synth.physical import amchirp
        synth_fn = getattr(amchirp, synthesis_fn)
        constants = config["constants"]
    else:
        raise ImportError(f"Unknown synthesis module: {synthesis_module}")

    return synth_fn, constants


def list_available_synthesizers() -> list:
    """
    List all available synthesizer types.

    Returns:
        List of available synthesizer type strings
    """
    return list(SYNTHESIZER_CONFIGS.keys())


def get_synth_description(synth_type: str) -> str:
    """
    Get human-readable description of synthesizer type.

    Args:
        synth_type: String identifier for synthesizer type

    Returns:
        Description string
    """
    config = get_synth_config(synth_type)
    return config.get("description", f"Synthesizer type: {synth_type}")
