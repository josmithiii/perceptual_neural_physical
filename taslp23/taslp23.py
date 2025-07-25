import functools
from kymatio.torch import TimeFrequencyScattering1D
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.neural import forward, loss
import torch.nn as nn
from pnp_synth.physical import ftm
from pnp_synth.synth_registry import get_synth_config, get_synthesis_function
import sklearn.preprocessing
import torch
import numpy as np

FOLDS = ["train", "test", "val"]
SAMPLES_PER_EPOCH = 512*50


mss_param = dict(
    max_n_fft=2048,
    num_scales=6,
    hop_lengths=None,
    mag_w=1.0,
    logmag_w=0.0,
    p=1.0,
)

def load_fold(synth_type, fold="full"):
    """Load DataFrame."""
    fold_dfs = {}
    csv_folder = os.path.join(os.path.dirname(__file__), "data")
    csv_name = "full_param_log.csv"
    csv_path = os.path.join(csv_folder, synth_type, csv_name)
    full_df = pd.read_csv(csv_path)
    full_df = full_df.sort_values(by="ID", ignore_index=False)
    assert len(set(full_df["ID"])) == len(full_df)
    if fold == "full":
        return full_df
    else:
        return full_df[full_df["fold"]==fold]


def pnp_forward_factory(scaler, logscale, synth_type):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. a synthesizer g (dynamic based on synth_type)
    3. a JTFS representation Phi
    """
    config = get_synth_config(synth_type)
    jtfs_params = config["jtfs_params"].copy()
    jtfs_params.update({
        "max_pad_factor": 1,
        "max_pad_factor_fr": 1,
        "pad_mode": 'zero',
        "pad_mode_fr": 'zero'
    })

    # Instantiate Joint-Time Frequency Scattering (JTFS) operator
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list")
    jtfs_operator.average_global = True

    Phi = functools.partial(S_from_x, jtfs_operator=jtfs_operator)

    g = functools.partial(x_from_theta, logscale=logscale, synth_type=synth_type)
    return functools.partial(
        forward.pnp_forward, Phi=Phi, g=g, scaler=scaler
    )


def scale_theta(logscale, synth_type):
    """
    Scale training set to [-1, 1], return values (NumPy array)
    and min-max scaler (sklearn object)
    """
    config = get_synth_config(synth_type)
    THETA_COLUMNS = config["theta_columns"]
    log_scale_columns = config["log_scale_columns"]

    # Load training set
    train_df = load_fold(synth_type, fold="train")
    # Fit scaler according to training set only
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    # Dynamic parameter processing for training data
    train_theta = []
    for column in THETA_COLUMNS:
        if not logscale and column in log_scale_columns:
            train_theta.append(10 ** train_df[column].values)
        else:
            train_theta.append(train_df[column].values)
    train_theta = np.stack(train_theta, axis=1)
    scaler.fit(train_theta)

    # Load whole dataset
    full_df = load_fold(synth_type, fold="full")

    # Transform whole dataset with scaler
    theta = []
    for column in THETA_COLUMNS:
        if not logscale and column in log_scale_columns:
            theta.append(10 ** full_df[column].values)
        else:
            theta.append(full_df[column].values)
    theta = np.stack(theta, axis=1)
    nus = scaler.transform(theta)
    return nus, scaler


def S_from_x(x, jtfs_operator):
    "Computes log-compressed Joint-Time Frequency Scattering."
    # Sx is a list of dictionaries
    Sx_list = jtfs_operator(x)

    # Convert to array
    Sx_array = torch.cat([path['coef'].flatten() for path in Sx_list])

    # apply "stable" log transformation
    # the number 1e3 is ad hoc and of the order of 1/mu where mu=1e-3 is the
    # median value of Sx across all paths
    log1p_Sx = torch.log1p(Sx_array*1e3)

    return log1p_Sx


def x_from_theta(theta, logscale, synth_type="ftm"):
    """Dynamic synthesizer based on synth_type."""
    synth_fn, constants = get_synthesis_function(synth_type)

    # Handle string synthesis special case (needs pos_ratio instead of logscale)
    if synth_type == "string":
        x = synth_fn(theta, 0.1, **constants)  # pos_ratio=0.1 for string
    else:
        x = synth_fn(theta, logscale, **constants)
    return x

def pnp_forward_factory_mss(scaler, logscale):
    """
    Computes S = (Phi o g o h^{-1})(nu) = (Phi o g)(theta) = Phi(x), given:
    1. a MinMax scaler h
    2. an FTM synthesizer g
    3. a JTFS representation Phi
    """

    g = functools.partial(x_from_theta, logscale=logscale)
    return functools.partial(
        forward.pnp_forward, Phi=MultiScaleSpectralLoss(), g=g, scaler=scaler
    )

class MultiScaleSpectralLoss(nn.Module):
    def __init__(
        self,
        max_n_fft=2048,
        num_scales=6,
        hop_lengths=None,
        mag_w=1.0,
        logmag_w=0.0,
    ):
        super().__init__()
        assert max_n_fft // 2 ** (num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        )
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]

    def forward(self, x):
        S = []
        for op in self.ops:
            S.append(op(x).flatten())
            print(op(x).shape)
        S = torch.cat(S)
        print("you passeed here?", S.shape)
        return S

class MagnitudeSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()
