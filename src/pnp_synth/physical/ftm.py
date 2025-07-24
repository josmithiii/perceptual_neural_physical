"""
This script implements Rabenstein's linear drum model
drum model with normalized side length ratio/ side length, in impulse form
"""
import numpy as np
import torch
from pnp_synth import utils

constants = {
    "x1": 0.4,
    "x2": 0.4,
    "h": 0.03,
    "l0": np.pi,
    "m1": 10,
    "m2": 10,
    "sr": 22050,
    "dur":2**16
}

constants_string = {
    "x": 0.1, # adapt to absolute position in meters
    "h": 0.03,
    "l0": np.pi,
    "m": 20,
    "sr": 22050,
    "dur":2**17
}



def rectangular_drum(theta, logscale, **constants):
    # Use the same device as the input tensor
    device = theta.device
    w11 = 10 ** theta[0] if logscale else theta[0]
    p = 10 ** theta[2] if logscale else theta[2]
    D = 10 ** theta[3] if logscale else theta[3]
    #theta
    tau11 = theta[1]
    alpha_side = theta[4]
    l0 = torch.tensor(constants['l0'], dtype=theta.dtype).to(device)

    l2 = l0 * alpha_side
    pi = torch.tensor(np.pi, dtype=theta.dtype).to(device)

    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (
        alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2
        + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)
    ) * (l0 / np.pi)**2
    T = c_sq # scalar
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi) **2

    EI = S ** 4

    mu = torch.arange(1, constants['m1'] + 1, dtype=theta.dtype).to(device) #(m1,)
    mu2 = torch.arange(1, constants['m2'] + 1, dtype=theta.dtype).to(device) #(m2,)
    dur = constants['dur']

    n = (mu[:,None] * pi / l0) ** 2 + (mu2[None,:] * pi / l2)**2 #(m1,m2)
    n2 = n ** 2
    K = torch.sin(mu[:,None] * pi * constants['x1']) * torch.sin(mu2[None,:] * pi * constants['x2']) #(m1,m2)

    beta = EI * n2 + T * n #(m1, m2)
    alpha = (d1 - d3 * n)/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))

    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants['sr'] / 2
    mode1_corr = constants['m1'] - max(torch.sum(mode_rejected, dim=0)) if constants['m1']-max(torch.sum(mode_rejected, dim=0))!=0 else constants['m1']
    mode2_corr = constants['m2'] - max(torch.sum(mode_rejected, dim=1)) if constants['m2']-max(torch.sum(mode_rejected, dim=1))!=0 else constants['m2']
    N = l0 * l2 / 4
    yi = (
        constants['h']
        * torch.sin(mu[:, None] * pi * constants['x1'])
        * torch.sin(mu2[None, :] * pi * constants['x2'])
        / omega #(m1, m2)
    )

    time_steps = torch.linspace(0, dur, dur, dtype=theta.dtype).to(device) / constants['sr'] #(T,)
    y = torch.exp(-alpha[:,:,None] * time_steps[None, None, :]) * torch.sin(
        omega[:,:,None] * time_steps[None,None,:]
    ) # (m1, m2, T)

    y = yi[:,:,None] * y #(m1, m2, T)
    y_full = y * K[:,:,None] / N
    #mode_rejected = mode_rejected.unsqueeze(2).repeat(1,1,y_full.shape[-1])
    y_full = y_full[:mode1_corr, :mode2_corr, :]
    #y_full[mode_rejected] -= y_full[mode_rejected]
    y = torch.sum(y_full, dim=(0,1)) #(T,)
    y = y / torch.max(torch.abs(y))

    return y


def rectangular_drum_batch(theta_batch, logscale, **constants):
    """
    Batch-compatible version of rectangular_drum. No speedup observed on MPS (batch size 200 or less).
    Jacobian computation is roughly 500x this cost according to Claude.

    Args:
        theta_batch: Tensor of shape [batch_size, 5] containing parameter vectors
        logscale: Whether to apply log scaling
        **constants: Same constants as rectangular_drum

    Returns:
        Tensor of shape [batch_size, dur] containing synthesized audio for each sample
    """
    # Use the same device as the input tensor
    device = theta_batch.device
    batch_size = theta_batch.shape[0]

    # Extract parameters with batch dimension - shape [batch_size]
    w11 = 10 ** theta_batch[:, 0] if logscale else theta_batch[:, 0]
    tau11 = theta_batch[:, 1]
    p = 10 ** theta_batch[:, 2] if logscale else theta_batch[:, 2]
    D = 10 ** theta_batch[:, 3] if logscale else theta_batch[:, 3]
    alpha_side = theta_batch[:, 4]

    # Constants - same for all samples
    l0 = torch.tensor(constants['l0'], dtype=theta_batch.dtype).to(device)
    pi = torch.tensor(np.pi, dtype=theta_batch.dtype).to(device)

    # Compute batch-wise intermediate values - shape [batch_size]
    l2 = l0 * alpha_side
    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (
        alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2
        + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)
    ) * (l0 / np.pi)**2
    T = c_sq  # scalar per batch
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi) **2
    EI = S ** 4

    # Mode indices - same for all samples
    mu = torch.arange(1, constants['m1'] + 1, dtype=theta_batch.dtype).to(device)  # [m1]
    mu2 = torch.arange(1, constants['m2'] + 1, dtype=theta_batch.dtype).to(device)  # [m2]
    dur = constants['dur']

    # Batch computation of modal properties
    # n computation - needs broadcasting for batch dimension
    # l2 has shape [batch_size], need to broadcast properly
    n_batch = (mu[None, :, None] * pi / l0) ** 2 + (mu2[None, None, :] * pi / l2[:, None, None])**2  # [batch_size, m1, m2]
    n2_batch = n_batch ** 2

    # K is the same for all samples since x1, x2 are constants
    K = torch.sin(mu[:, None] * pi * constants['x1']) * torch.sin(mu2[None, :] * pi * constants['x2'])  # [m1, m2]
    K_batch = K[None, :, :].expand(batch_size, -1, -1)  # [batch_size, m1, m2]

    # Batch computation of modal parameters
    beta_batch = EI[:, None, None] * n2_batch + T[:, None, None] * n_batch  # [batch_size, m1, m2]
    alpha_batch = (d1[:, None, None] - d3[:, None, None] * n_batch) / 2  # [batch_size, m1, m2]
    omega_batch = torch.sqrt(torch.abs(beta_batch - alpha_batch**2))  # [batch_size, m1, m2]

    # Mode rejection based on Nyquist frequency - computed per batch sample
    mode_rejected_batch = (omega_batch / 2 / pi) > constants['sr'] / 2  # [batch_size, m1, m2]

    # For simplicity, use the same mode correction for all batch samples
    # (More sophisticated version could compute per-sample corrections)
    mode_rejected_any = torch.any(mode_rejected_batch, dim=0)  # [m1, m2]
    mode1_corr = constants['m1'] - max(torch.sum(mode_rejected_any, dim=0)) if constants['m1']-max(torch.sum(mode_rejected_any, dim=0))!=0 else constants['m1']
    mode2_corr = constants['m2'] - max(torch.sum(mode_rejected_any, dim=1)) if constants['m2']-max(torch.sum(mode_rejected_any, dim=1))!=0 else constants['m2']

    # Batch computation of yi
    N_batch = l0 * l2 / 4  # [batch_size]
    yi_batch = (
        constants['h']
        * torch.sin(mu[None, :, None] * pi * constants['x1'])
        * torch.sin(mu2[None, None, :] * pi * constants['x2'])
        / omega_batch  # [batch_size, m1, m2]
    )

    # Time evolution - this is the expensive part
    time_steps = torch.linspace(0, dur, dur, dtype=theta_batch.dtype).to(device) / constants['sr']  # [dur]

    # Batch time evolution computation - shape [batch_size, m1, m2, dur]
    y_time = torch.exp(-alpha_batch[:, :, :, None] * time_steps[None, None, None, :]) * torch.sin(
        omega_batch[:, :, :, None] * time_steps[None, None, None, :]
    )

    # Apply yi scaling and K weighting
    y_batch = yi_batch[:, :, :, None] * y_time  # [batch_size, m1, m2, dur]
    y_full_batch = y_batch * K_batch[:, :, :, None] / N_batch[:, None, None, None]

    # Apply mode correction (same for all batch samples)
    y_full_batch = y_full_batch[:, :mode1_corr, :mode2_corr, :]

    # Sum over modes and normalize
    y_final = torch.sum(y_full_batch, dim=(1, 2))  # [batch_size, dur]

    # Normalize each sample independently
    y_max = torch.max(torch.abs(y_final), dim=1, keepdim=True)[0]  # [batch_size, 1]
    y_final = y_final / y_max

    return y_final

def physics2percep(S4, T, d1, d3, l, lm):
    c2 = T/lm
    sigma1 = d3/(2*lm) * (np.pi/l)**2 - d1/(2*lm)
    tau1 = 1/sigma1
    w1 = np.sqrt((S4-d3**2/(4*lm**2))*(np.pi/l)**4 + (c2 + d1*d3/(2*lm**2)) * (np.pi/l)**2 - d1**2/(4*lm**2))
    p = d3 * tau1 / (2*lm) * (np.pi/l)**2
    D = np.sqrt(S4 * (np.pi/l)**4 - (p*sigma1)**2) / w1
    return w1, tau1, p, D

def percep2physics(w1, tau1, p, D, l, lm):
    # convert perceptual parameters to PDE parameters
    #d1 = 2 * (1 - p * lm * (l/np.pi)**2 ) / tau1 # wrong
    d3 = 2 * p * lm * l**2 / (tau1 * np.pi**2)
    d1 = -2 * lm / tau1 + d3 * (np.pi/l)**2
    S4 = (l/np.pi)**4 * ((D*w1)**2 + (p/tau1)**2)
    c2 = (l/np.pi)**2 * (w1**2 * (1-D**2) + (1-p**2)/tau1**2)
    return d1, d3, S4, c2

# theta = {w1,tau1, p, D, lm, ell}
def linearstring_percep(theta, logscale, **constants_string):
    device = utils.get_device()
    # convert omega, tau, p, D into S, c, d1, d3
    w11 = 10 ** theta[0] if logscale else theta[0]
    p = 10 ** theta[2] if logscale else theta[2]
    D = 10 ** theta[3] if logscale else theta[3]
    #theta
    tau11 = theta[1]
    lm = theta[4]
    ell = theta[5]
    pi = torch.tensor(np.pi, dtype=torch.float32).to(device)
    dur = constants_string['dur']

    d1, d3, S4, c2 = percep2physics(w11, tau11, p, D, ell, lm)
    d1 = abs(d1)
    EI = S4 * lm
    Ts0 = c2 * lm

    mu = torch.arange(1, constants_string["m"] + 1).to(device)
    n = (mu * pi / ell) ** 2
    n2 = n ** 2
    K = torch.sin(mu * pi * constants_string["x"])

    beta = EI * n2 + Ts0 * (-n) #(m)
    alpha = (d1 + d3 * n)/(2*lm) # nonlinear
    omega = torch.sqrt(torch.abs(beta/(lm) - alpha**2))
    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants_string['sr'] / 2
    mode_corr = constants_string['m'] - torch.sum(mode_rejected)

    N = ell / 2
    yi = (
        constants_string['h']
        * torch.sin(mu * pi * constants_string["x"]) #this should be the listening position
        / omega #(mode)
    )

    time_steps = torch.linspace(0, dur, dur).to(device) / constants_string['sr'] #(T,)

    y = torch.exp(-alpha[:,None] * time_steps[ None, :]) * torch.sin(
        omega[:,None] * time_steps[None,:]
    ) # (m, T)

    y = yi[:, None] * y #(m, T)
    y_full = y * K[:,None] / N
    y_full = y_full[:mode_corr, :]
    y = torch.sum(y_full, dim=0) #(T,)
    y = y / torch.max(torch.abs(y))


    return y



#theta:{EI, T, d1, d3, lm, ell}
def linearstring_physics(theta, pos_ratio, **constants_string):
    """
    unlike the convention in rabenstein's paper. d3 is always positive, so alpha=(d1+d3*n)/(2*lm)
    beta = EI n2 + Ts0 n (positive sign here)
    """
    device = utils.get_device()
    # convert omega, tau, p, D into S, c, d1, d3
    EI = theta[0]
    Ts0 = 10 ** theta[1]
    d1 = theta[2]
    d3 = theta[3]
    lm = 10 ** theta[4]
    ell = 10 ** theta[5]
    pi = torch.tensor(np.pi, dtype=torch.float32).to(device)
    dur = constants_string['dur']


    mu = torch.arange(1, constants_string["m"] + 1).to(device)
    n = (mu * pi / ell) ** 2
    n2 = n ** 2
    K = torch.sin(mu * pi * pos_ratio)

    beta = EI * n2 + Ts0 * n #(m)
    alpha = (d1 + d3 * n)/(2*lm) # nonlinear
    # TODO: there should be constraint in how alpha should be, in case it exceeds beta!!!

    omega = torch.sqrt(beta/(lm) - alpha**2)
    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants_string['sr'] / 2
    mode_corr = constants_string['m'] - torch.sum(mode_rejected)
    if torch.sum(torch.isnan(omega)) > 0 or torch.min(omega) > 1500 * 2 * np.pi:
        return "exceeded pitch range"
    else:
        N = ell / 2
        yi = (
            constants_string['h']
            * torch.sin(mu * pi * pos_ratio) #this should be the listening position
            / omega #(mode)
        )

        time_steps = torch.linspace(0, dur, dur).to(device) / constants_string['sr'] #(T,)

        y = torch.exp(-alpha[:,None] * time_steps[ None, :]) * torch.sin(
            omega[:,None] * time_steps[None,:]
        ) # (m, T)

        y = yi[:, None] * y #(m, T)
        y_full = y * K[:,None] / N
        y_full = y_full[:mode_corr, :]
        y = torch.nansum(y_full, dim=0) #(T,)
        y = y / torch.max(torch.abs(y))


        return y
