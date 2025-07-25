# Joint Time-Frequency Scattering (JTFS) Transform

This document provides comprehensive implementation documentation for the Joint Time-Frequency Scattering transform, extracted from the `timefrequency_scattering1d` function.

## Table of Contents

- [Overview](#overview)
- [Frequential Scattering](#frequential-scattering)
  - [Variables](#variables)
- [Subsampling, Padding](#subsampling-padding)
  - [aligned=True](#alignedtrue)
  - [out_3D=True](#out_3dtrue)
  - [log2_F](#log2_f)
- [Debug Tips](#debug-tips)
- [Implementation Details](#implementation-details)
- [Exhaustive Diagrams](#exhaustive-diagrams)
  - [_joint_lowpass()](#_joint_lowpass)
  - [X. aligned==True](#x-alignedtrue)
  - [Y. aligned==False](#y-alignedfalse)
  - [Example w/ `True, True`](#example-w-true-true)
- [Final Algorithm](#final-algorithm)
- [JTFS in the Perceptual-Neural-Physical (PNP) Pipeline](#jtfs-in-the-perceptual-neural-physical-pnp-pipeline)
  - [Complete Pipeline Overview](#complete-pipeline-overview)
  - [JTFS Configuration (TASLP23)](#jtfs-configuration-taslp23)
  - [Computational Characteristics](#computational-characteristics)
  - [Implementation Details](#implementation-details-1)
  - [Research Applications](#research-applications)
  - [Alternative Approaches](#alternative-approaches)

## Overview

Main function implementing the Joint Time-Frequency Scattering transform.

Below is implementation documentation for developers.

## Frequential Scattering

### Variables

Explanation of variable naming and roles. For `scf`'s attributes see full Attributes docs in `TimeFrequencyScattering1D`.

#### subsample_equiv_due_to_pad
Equivalent amount of subsampling due to padding, relative to `J_pad_frs_max_init`. Distinguishes differences in input lengths (to `_frequency_scattering()` and `_joint_lowpass()`) due to padding and conv subsampling. This is needed to tell whether input is *trimmed* (pad difference) or *subsampled*.

#### "conv subsampling"
Refers to `subsample_fourier()` after convolution (as opposed to equivalent subsampling due to padding). i.e. "actual" subsampling. Refers to `total_conv_stride_over_U1`.

#### total_conv_stride_over_U1
See above. Alt name: `total_conv_stride_fr`; `over_U1` seeks to emphasize that it's the absolute stride over first order coefficients that matters rather than equivalent/relative quantities w.r.t. padded etc

#### n1_fr_subsample
Amount of conv subsampling done in `_frequency_scattering()`.

#### total_downsample_fr
Total amount of subsampling, conv and equivalent, relative to `J_pad_frs_max_init`. Conceptual quantity.

## Subsampling, Padding

Controlled by `aligned`, `out_3D`, `average_fr`, `log2_F`, and `sampling_psi_fr` & `sampling_phi_fr`.

- `freq` == number of frequential rows (originating from U1), e.g. as in `(n1_fr, freq, time)` for joint slice shapes per `n2` (with `out_3D=True`).
- `n1_fr` == number of frequential joint slices, or `psi1_f_fr_*`, per `n2`
- `n2` == number of `psi2` wavelets, together with `n1_fr` controlling total number of joint slices

### aligned=True

Imposes:
- `total_conv_stride_over_U1` to be same for all joint coefficients. Otherwise, row-to-row log-frequency differences, `dw,2*dw,...`, will vary across joint slices, which breaks alignment.
- `sampling_psi_fr=='resample'`: center frequencies must be same
- `sampling_phi_fr`: not necessarily restricted, as alignment is preserved under different amounts of frequential smoothing, but bins may blur together (rather unblur since `False` is finer); for same fineness/coarseness across slices, `True` is required.

#### average_fr=False
- Additionally imposes that `total_conv_stride_over_U1==0`, since `psi_fr[0]['j'][0] == 0` and `_joint_lowpass()` can no longer compensate for variable `j_fr`.
- Enforces separate `total_conv_stride_over_U1` for frequential lowpass coeffs: `phi_t * phi_f`, `psi_t * phi_f` will have their own common stride. This avoids significant redundancy while still complementing joint coeffs with lowpass information.

#### out_3D=True
Additionally imposes that all frequential padding is the same (maximum), since otherwise same `total_conv_stride_over_U1` yields variable `freq` across `n2`.

#### average_fr_global=True
Stride logic in `_joint_lowpass()` is relaxed since all U1 are collapsed into one point; will no longer check `total_conv_stride_over_U1`

### out_3D=True

Imposes:
- `freq` to be same for all `n2` (can differ due to padding or convolutional stride)

### log2_F

- Larger -> smaller `freq`
- Larger -> greater `max_subsample_before_phi_fr`
- Larger -> greater `J_pad_frs_max` (only if `log2_F > J_fr`)

## Debug Tips

Check following scf attributes:
- N_frs
- J_pad_frs, J_pad_frs_max_init
- ind_end_fr, ind_end_fr_max

## Implementation Details

In regular scattering, `log2_T` controls maximum total subsampling upon padded input; if padding beyond nextpow2, will unpad more, not subsample more

Any reason for `total_conv_stride_over_U1` to not equal `log2_F`?
- in `aligned=True` && `out_3D=False`, we pad variably, so min padded cannot stride same as max padded else we may collapse `freq` to < 0. Then what decides maximum stride?

Note, if `log2_F` is 1 less than global averaging we get `freq=2` for max padded case, then this stride on min padded case would yield < 0. Thus we restrict maximum stride based on min padded case, and total strides (psi + phi) for all `n1_fr` must equal `log2_F - subsample_equiv_due_to_pad_min`.

- in `aligned=True` && `out_3D=True` we always pad to maximum, thus we can (and should) have `total_conv_stride_over_U1 == log2_F`.

Since we must have same `total_conv_stride_over_U1` for all `n1_fr`, in variably padded case we'll necessarily get variable `freq` per `n1_fr`, forbidding 3D concatenation (and which is why `out_3D=True` always pads same).

In `aligned=False` we don't care about enforcing the same stride, so we subsample maximally for each padding - that is, as much as the filters will permit. The only restriction is, still, to not end up with <= 0 `freq`.
- "Pad more -> unpad mode" still holds in fr scattering, so `total_downsample_fr_max` is set relative to `nextpow2(N_frs_max)`. That is, shorter `freq` due to the combination of padding less, and conv stride (subsampling), is the same for all paddings, and for greater `J_pad_fr` we simply unpad more.

What is `log2_F` set relative to, and can `total_downsample_fr_max == log2_F`?
- `log2_F` is set by the user indirectly via `F` to control amount of imposed frequency transposition invariance, via temporal support of lowpass.
- `log2_F` controls max subsampling we can do after conv with `phi_fr`.

What does this mean for inputs (to `phi_fr`) of varying length - namely, length lesser than maximum (which is the length `phi_fr` was originally sampled at, and length at which it could be subsampled by `2**log2_F`)?
- With `resample_=True`, we can still subsample by `log2_F` without alias - restrictions will come from elsewhere (`aligned`, `freq` <= 0, etc).
- With `resample_=False`, allowed subsampling is less than `log2_F`, and this is accounted for alongside existing constraints.

Thus: account for `phi_fr`'s `j` at any length along other constraints.
- `log2_F` cannot exceed `nextpow2(N_frs_max)`; can `total_downsample_fr_max` exceed it? For max padded case, whose pad length might exceed `nextpow2(N_frs_max)`, we can still total subsample by `log2_F` at most (`average_fr=True`; note this quantity is unused for `False`); the rest will "unpad more".

Thus, `total_downsample_fr_max <= log2_F`. Can it be <? The *actual* subsampling we do can be, but it'll be due to factors unrelated to subsampling permitted by `phi_fr`.

Thus, `total_downsample_fr_max == log2_F`. (effectively by definition)

- However we still need a quantity that denotes *actual* maximum subsampling we'll do at any pad length, given configuration (`aligned` etc). Call it `total_downsample_fr_max_adj`.
  - Differs from `total_downsample_fr_max` when `aligned`: total conv stride is restricted by min padded case.

## Exhaustive Diagrams

### _joint_lowpass()

#### Conditions

`phi_fr[subsample_equiv_due_to_pad][n1_fr_subsample]` indexing for all cases. What differs is subsampling afterwards, determined by:

**C1)** `j` of the `phi_fr`
- depends on `sampling_phi_fr`

**C2)** whether `freq` should be same for all `n2, n1_fr`
- depends on `out_3D`

**C3)** whether `total_conv_stride_over_U1` should be same for all `n2, n1_fr`
- depends on `aligned`

**C4)** whether `freq` <= 0
- independent of `aligned, out_3D, resample_*`
- must enforce total subsampling (equivalent and due to stride) does not exceed `pad_fr`
- indirect dependence on `sampling_phi_fr` where, in `True`, we can subsample by more than `log2_F`, which is accounted for by `max_subsample_before_phi_fr`

#### Definitions

**D1)** `total_downsample_fr` refers to total subsampling (equivalent due to padding less, and due to convolutional stride) relative to `J_pad_frs_max_init`. That is, `total_downsample_fr == freq / J_pad_frs_max_init`, where `freq` is length of frequential dimension *before* unpadding. It is also == `lowpass_subsample_fr + n1_fr_subsample + subsample_equiv_due_to_pad`.

**D2)** `log2_N_fr == nextpow2(N_frs_max)` is the maximum possible `total_conv_stride_over_U1` for any configuration. (Exceeding would imply `freq` <0 0 over unpadded `N_frs_max`).
- `log2_N_fr + diff == J_pad_frs_max_init` is the maximum possible `total_downsample_fr` for any configuration.
- `log2_F <= log2_N_fr` for all configurations.

**D3)** `sampling_psi_fr == True` shall refer to `== 'resample'`, and `False` to `'recalibrate'` or `'exclude'`. Likewise for phi (except there's no `'exclude'`).

#### Observations

**O1)** With `aligned==True`, `lowpass_subsample_fr` must be a deterministic function of `n1_fr_subsample` (i.e. have same value at `n1_fr` for all `n2`), otherwise `total_conv_stride_over_U1` will vary over `n2`.

**O2)** With `out_3D==True`, by similar logic, `total_downsample_fr` must be same for all `n1_fr, n2` to get same `freq`.

### X. aligned==True

#### A. out_3D==True

**sampling_psi_fr, sampling_phi_fr:**

**1. True, True:**

`lowpass_subsample_fr == log2_F - n1_fr_subsample`. Because:

a. `lowpass_subsample_fr` cannot exceed `log2_F` because `log2_F` is the maximum subsampling factor of any `phi_fr`.

b. `lowpass_subsample_fr` can be less than `log2_F` because must subsample less to get same `freq` with different `n1_fr_subsample`

c. `lowpass_subsample_fr` and `n1_fr_subsample` must sum to the same (`total_conv_stride_over_U1`)

d. `total_conv_stride_over_U1` cannot exceed `log2_F` because minimum `n1_fr_subsample == 0`, and max `lowpass_subsample_fr == log2_F`.

e. `phi_fr` will be subsample-able by `log2_F` for any `pad_fr` (`J_pad_fr` will be restricted by `max_subsample_before_phi_fr` to ensure this).

f. `freq` <= 0 isn't possible because we always have `pad_fr == J_pad_frs_max == J_pad_frs_max_init` (or equivalently `pad_fr == J_pad_frs_max <= J_pad_frs_max_init` if not `True, True`, where `J_pad_frs_max >= log2_N_fr`, and `log2_F <= log2_N_fr` (D2)).

g. `max(..., 0)` isn't necessary since `n1_fr_subsample <= log2_F` is enforced (to avoid aliased `phi_fr`, note a; `subsample_equiv_due_to_pad + n1_fr_subsample` may exceed `log2_F`, but that won't require `phi_fr` subsampled beyond `log2_F` per `_, True`).

h. `lowpass_subsample_fr`, `n1_fr_subsample`, and `subsample_equiv_due_to_pad` must sum to the same value (`total_downsample_fr`) for all `n2, n1_fr` (to get same `freq`). Since `subsample_equiv_due_to_pad` is same for all `n2, n1_fr`, only source of variability is `n1_fr_subsample`.

`total_downsample_fr == log2_F`.

i. `total_downsample_fr` cannot exceed `log2_F` due to `pad_fr == J_pad_frs_max == J_pad_frs_max_init`.

`total_conv_stride_over_U1 == log2_F`

d+. `lowpass_subsample_fr == log2_F` will happen along `n1_fr_subsample == 0`. Per c, `total_conv_stride_over_U1` hence also cannot be less, and must `== log2_F`.

**2. False, True:**

Same as 1. Because:

a. Same as 1's. `J_pad_frs_max < J_pad_frs_max_init` is possible, but 1e applies.

b,c,d,e,g,h: same as 1's.

f. Same as 1's (`pad_fr == J_pad_frs_max >= log2_N_fr`).

`total_downsample_fr == log2_F + diff`, where `diff == J_pad_frs_max_init - J_pad_frs_max`.

i. `total_downsample_fr` can exceed `log2_F` because we can have `pad_fr == J_pad_frs_max < J_pad_frs_max_init`, and `_, True` will keep the `< 2**J_pad_frs_max_init` length `phi_fr`'s subsampling factor at `log2_F` (i.e. 1e).

`total_conv_stride_over_U1 == log2_F`

d+. Same as 1's (also see 2f).

**3. True, False:**

`lowpass_subsample_fr == (log2_F - diff) - n1_fr_subsample`. Because:

a,b,c,h: same as 1's.

d. same as 1's except `log2_F` -> `log2_F - diff` (see 3e).

e. `phi_fr` will not be subsample-able by `log2_F` for all `pad_fr`s. `phi_fr` with `j=log2_F` is sampled at `2**J_pad_frs_max_init`, and subsequently has `j<log2_F` (see 3f). That is, `_, False` makes `phi_fr`'s `j` directly depend on input length, so both `n1_fr_subsample` and `subsample_equiv_due_to_pad` lower `j`. `_, False` (or `False, _`) enables `J_pad_frs_max < J_pad_frs_max_init`.

f. same as 1's (+2's reasoning).

g. same as 1's, except `n1_fr_subsample <= log2_F - diff` is enforced per c and 3d+.

`total_downsample_fr == log2_F`. Because,

i. Per 3e, exceeding `log2_F` would mean subsampling `phi_fr` beyond `log2_F`.

`total_conv_stride_over_U1 == log2_F - diff`

d+. Same as 1's, except `log2_F` -> `log2_F - diff` per 3e.

**4. False, False:**

Same as 3. `False, _` only affects `J_pad_fr` (but `pad_fr == J_pad_frs_max`, always) and `J_pad_frs_max` (and 3 accounts for this).

```
lowpass_subsample_fr == phi_fr_sub_at_max - n1_fr_subsample
phi_fr_sub_at_max == phi_fr['j'][J_pad_frs_max] <= log2_F
# ^ maximum "realizable" subsampling after `phi_fr`
```

accounts for 1-4.
- `phi_fr_sub_at_max == log2_F` for 1,2 and we reduce to `lowpass_subsample_fr == log2_F - n1_fr_subsample`
- `phi_fr_sub_at_max == log2_F - diff` for 3,4.

```
total_downsample_fr == log2_F + (diff - j_diff)
j_diff == phi_fr['j'][J_pad_frs_max_init] - phi_fr['j'][J_pad_frs_max]
```

accounts for 1-4.
- 1: `J_pad_frs_max == J_pad_frs_max_init` -> `diff == j_diff == 0` -> `total_downsample_fr == log2_F`
- 2: `J_pad_frs_max <= J_pad_frs_max_init` -> `diff >= 0`, `j_diff == 0` -> `total_downsample_fr == log2_F + diff`
- 3: `J_pad_frs_max <= J_pad_frs_max_init` -> `diff == j_diff >= 0` -> `total_downsample_fr == log2_F`
- 4: same as 3.

```
total_conv_stride_over_U1 == log2_F - j_diff
                          == phi_fr_sub_at_max
```

accounts for 1-4.
- `== log2_F` for 1,2, and `== log2_F - diff` for 3,4.

#### B. out_3D==False

^1: `pad_fr == J_pad_frs_max` no longer always holds
^2: same `freq` for all `n1_fr, n2` no longer required

**sampling_psi_fr, sampling_phi_fr:**

**1. True, True**

`lowpass_subsample_fr == min(log2_F, J_pad_frs_min) - n1_fr_subsample`. Because:

a. == A1a.

b. `lowpass_subsample_fr` does not need to be less than `log2_F` due to B^2.

c. == A1c. This overrides b.

d. differs from A1d (in "max `lowpass_subsample_fr`"). See B1d+.

e. == A1e.

f. `freq` <= 0 is possible without `min` per B^1. `freq` <= 0 will occur upon `lowpass_subsample_fr > pad_fr`, so max `== pad_fr`. (namely `== J_pad_frs_min` per c).

g. == A1g, except `n1_fr_subsample <= min(log2_F, J_pad_frs_min)` is enforced per c and B1d+ (or to avoid `lowpass_subsample_fr < 0`)

h. N/A per B^2.

`total_downsample_fr == (total_conv_stride_over_U1 + subsample_equiv_due_to_pad)`

(this holds by definition; can plug expressions, but this is cleaner)

i. `total_downsample_fr` can exceed `log2_F` because we can have `pad_fr < J_pad_frs_max == J_pad_frs_max_init` (or equivalently `pad_fr < J_pad_frs_max <= J_pad_frs_max_init` for not `True, True`). (`subsample_equiv_due_to_pad == J_pad_frs_max_init - pad_fr` is the generalization of `diff` (see e.g. A2g))

`total_conv_stride_over_U1 == min(log2_F, J_pad_frs_min)`.

d+. Smallest conv stride is determined from `J_pad_frs_min` such that `total_downsample_fr <= J_pad_frs_min`, i.e. sum of `lowpass_subsample_fr`, `n1_fr_subsample`, and `subsample_equiv_due_to_pad` is `<= J_pad_frs_min`. Smallest conv stride will occur at `n1_fr_subsample==0`, equal to `lowpass_subsample_fr`'s maximum at `J_pad_frs_min` (which is `log2_F` if `log2_F <= J_pad_frs_min`, else `J_pad_frs_min`). c then forbids exceeding this (and can't lower since this is min).

**2. False, True:**

Same as 1. `J_pad_frs_max < J_pad_frs_max_init` doesn't change any expression (but values within expressions might).

**3. True, False:**

`lowpass_subsample_fr == (log2_F - diffmin) - n1_fr_subsample`, where `diffmin == J_pad_frs_max_init - J_pad_frs_min`. Because:

a,b,c,d,h: same as 1's.

e. == A3e.

f. `freq` <= 0 is prevented by `total_conv_stride_over_U1 ==` `log2_F - diffmin <= J_pad_frs_min`.

g. same as A1's except `n1_fr_subsample <= log2_F - diffmin` is enforced per c and 3d+.

`total_downsample_fr == (total_conv_stride_over_U1 + subsample_equiv_due_to_pad)`

i. == A3i. Peaks when `pad_fr==J_pad_frs_min`, at `log2_F`.

`total_conv_stride_over_U1 == log2_F - diffmin`.

d+. == 1d+, except `lowpass_subsample_fr`'s maximum at `J_pad_frs_min` is `log2_F - diffmin`, and we always have `log2_F - diff <= J_pad_frs_min`. (Since `log2_F` occurs at J_pad_frs_max_init, if `log2_F == log2_N_fr` (where `log2_N_fr <= J_pad_frs_max_init`), then at `J_pad_frs_min` it's `log2_N_fr - diffmin <= J_pad_frs_max_init - diffmin` `== J_pad_frs_min`. If `log2_F < log2_N_fr`, then `log2_F - diffmin < J_pad_frs_min` (instead of <=).)

**4. False, False:**

Same as 3. `False, _` only affects `pad_fr` via `J_pad_fr` (and 3 accounts for this).

```
lowpass_subsample_fr == (min(phi_fr_sub_at_min, J_pad_frs_min) - n1_fr_subsample)
phi_fr_sub_at_min == phi_fr['j'][k_at_min] <= log2_F;
k_at_min = J_pad_frs_max_init - J_pad_frs_min
# ^ subsampling after `phi_fr` in min-padded case
```

accounts for 1-4.
- `phi_fr_sub_at_min == log2_F` for 1,2 and we reduce to `lowpass_subsample_fr == min(log2_F, J_pad_frs_min) - n1_fr_subsample`
- `phi_fr_sub_at_min == log2_F - diffmin <= J_pad_frs_min` for 3,4.

```
total_downsample_fr == (subsample_equiv_due_to_pad + total_conv_stride_over_U1)
```

True by definition, but expressions for involved variables will vary.

```
total_conv_stride_over_U1 == phi_fr_sub_at_min
```

accounts for 1-4. Follows d+'s logic.

Observe that in all cases, A and B, `lowpass_subsample_fr` results from satisfying C1,C3,C4, where we consider `total_conv_stride_over_U1` in min padded case, and set that as maximum permissible `lowpass_subsample_fr`.

### Y. aligned==False

This relaxes c, d, B^1 in all of X:

c. `lowpass_subsample_fr` and `n1_fr_subsample` no longer need to sum to the same value (`total_conv_stride_over_U1`)

d. `total_conv_stride_over_U1` can exceed `phi_fr_sub_at_max` per c.

and applies B^1:

B^1. `pad_fr == J_pad_frs_max` for all `n1_fr, n2` is only needed with `aligned and out_3D`.

`lowpass_subsample_fr == min(log2_F, pad_fr) - n1_fr_subsample`.

#### A. out_3D==True

**sampling_psi_fr, sampling_phi_fr:**

**1. True, True:**

`lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample`. This holds by definition; see j. Rationale:

a,b,e: same as X.A.1's.

f. == XB1f, except max is now `pad_fr` (varies over `n2`) rather than `J_pad_frs_min` (fixed) per relaxing c.

g. See j.

h. == XA1h, except `subsample_equiv_due_to_pad` varies. Thus `total_conv_stride_over_U1` must compensate.

`total_downsample_fr == (total_conv_stride_over_U1 + subsample_equiv_due_to_pad)`

i. == XB1i. See j.

`total_conv_stride_over_U1 = (min(log2_F, J_pad_frs_max) - (J_pad_frs_max - pad_fr))`

j. max `total_conv_stride_over_U1` is determined in `J_pad_frs_max` case; for each `pad_fr` we can still subsample by up to `log2_F`, so lesser `pad_fr` -> greater `total_downsample_fr`, yielding different `freq`. To compensate, adjust `total_conv_stride_over_U1` relative to max padded case, to subsample less w/ lesser `pad_fr`. Expanding `lowpass_subsample_fr`:

`lowpass_subsample_fr = (min(log2_F, J_pad_frs_max) - (J_pad_frs_max - pad_fr) - n1_fr_subsample)`

Thus we enforce `n1_fr_subsample <= (min(log2_F, J_pad_frs_max) - (J_pad_frs_max - pad_fr))` to avoid `lowpass_subsample_fr < 0`.

**2. False, True**

Same as 1. See XB2.

**3. True, False**

`lowpass_subsample_fr = (min(log2_F, J_pad_frs_max) - subsample_equiv_due_to_pad - n1_fr_subsample)`. This is simply 1 with `J_pad_frs_max_init - pad_fr`, and `min` since `J_pad_frs_max < log2_F` is possible. It then follows:

`n1_fr_subsample <= (min(log2_F, J_pad_frs_max) - subsample_equiv_due_to_pad)`

`total_conv_stride_over_U1 = (min(log2_F, J_pad_frs_max) - subsample_equiv_due_to_pad)`

**4. False, False**

Same as 3.

```
lowpass_subsample_fr == (min(phi_fr_sub_at_max, J_pad_frs_max) -
                         (J_pad_frs_max - pad_fr) - n1_fr_subsample)
phi_fr_sub_at_max == phi_fr['j'][J_pad_frs_max]
```

accounts for 1-4.

```
total_conv_stride_over_U1 = (min(phi_fr_sub_at_max, J_pad_frs_max) -
                             (J_pad_frs_max - pad_fr))`
total_downsample_fr = (total_conv_stride_over_U1 +
                       subsample_equiv_due_to_pad)
```

### Example w/ `True, True`

```
J_pad_frs_min = 7
J_pad_frs_max = 10
J_pad_frs_max_init = 11
log2_F = 5
j1_fr_max = 6

pad_fr == J_pad_frs_min == 7:
    subsample_equiv_due_to_pad = 11 - 7 = 4
    J_pad_frs_max - pad_fr = 3

    j1_fr == 6:
        n1_fr_subsample = 5 - 3 = 2
        lowpass_subsample_fr = 5 - 3 - 2 = 0
        total_conv_stride_over_U1 = 2 + 0 = 2
        total_downsample_fr = 2 + 4 = 6

    j1_fr == 0:
        n1_fr_subsample = 0
        lowpass_subsample_fr = 5 - 3 - 2 = 2
        total_conv_stride_over_U1 = 0 + 2 = 2
        total_downsample_fr = 2 + 4 = 6

pad_fr == J_pad_frs_max == 10:
    subsample_equiv_due_to_pad = 1
    j1_fr == 6:
        n1_fr_subsample = 5
        lowpass_subsample_fr = 0
        total_conv_stride_over_U1 = 5 + 0 = 5
        total_downsample_fr = 5 + 1 = 6

    j1_fr == 0:
        n1_fr_subsample = 0
        lowpass_subsample_fr = 5
        total_conv_stride_over_U1 = 0 + 5 = 5
        total_downsample_fr = 5 + 1 = 6
```

#### B. out_3D==False

XB^2 relaxed.

This leaves only one condition: "not `freq` <= 0". We can now subsample maximally at every stage:

```
lowpass_subsample_fr = j_phi - n1_fr_subsample
total_conv_stride_over_U1 = j_phi
total_downsample_fr = j_phi + subsample_equiv_due_to_pad
n1_fr_subsample <= j_phi
```

accounts for 1-4.

## Final Algorithm

Accounting for all of X and Y:

```python
k = subsample_equiv_due_to_pad
if aligned:
    if out_3D:
        total_conv_stride_over_U1 = phi_fr['j'][k]
    else:
        k_at_min = J_pad_frs_max_init - J_pad_frs_min
        total_conv_stride_over_U1 = min(phi_fr['j'][k_at_min], J_pad_frs_min)
else:
    if out_3D:
        total_conv_stride_over_U1 = (min(phi_fr['j'][k], J_pad_frs_max) -
                                     (J_pad_frs_max - pad_fr))
    else:
        total_conv_stride_over_U1 = phi_fr['j'][k]

n1_fr_subsample = min(j1_fr, total_conv_stride_over_U1)
lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
total_downsample_fr = n1_fr_subsample + lowpass_subsample_fr + k
```

## JTFS in the Perceptual-Neural-Physical (PNP) Pipeline

This project uses JTFS as a key component in a **Perceptual-Neural-Physical (PNP) synthesis pipeline** for training neural networks to perform inverse synthesis (audio → physical parameters).

### Complete Pipeline Overview

The project implements a three-stage pipeline for neural audio parameter estimation:

#### Stage 1: Forward Synthesis + Jacobian Computation (`02_compute_pnp_jacobian.py`)
```
Physical Parameters → FTM Synthesis → Audio → JTFS → Coefficients + Jacobians
     [ω,τ,p,D,α]    →   [65,536]   → [~20,762]  →    S.npy + J.npy
```

#### Stage 2: Riemannian Metric Computation (`12_compute_lmastep_fast.py`)
```
Jacobians → M = J^T J → Eigenvalues → Riemannian Weights
  J.npy   →   M.npy   →   sigma.npy  →   PNP Loss Weights
```

#### Stage 3: Neural Network Training (`05_train_effnet_pnploss.py`)
```
Audio Features + Parameters + Weights → EfficientNet → Parameter Predictions
    JTFS/CQT    +   Ground Truth   +   M matrices   →    θ̂ = NN(audio)
```

The complete pipeline enables:

1. **Forward modeling**: Generate large datasets of (audio, parameters) pairs with perceptual features
2. **Geometric analysis**: Compute Riemannian metrics for proper loss weighting
3. **Inverse synthesis**: Train neural networks for parameter estimation from audio

### JTFS Configuration (TASLP23)

The JTFS transform is configured for comprehensive perceptual analysis:

```python
# From taslp23.py and utils.py
J = 13              # 13 temporal scales (12 octaves coverage)
Q = (12, 1)         # 12 filters/octave (1st order), 1 filter/octave (2nd order)
shape = (2**16,)    # 65,536 samples (matches FTM output)
F = 2               # Frequency averaging factor
T = 2**16           # Time support
sr = 22050          # Sample rate (from FTM synthesis)
```

This creates **289 scattering paths** across multiple temporal and frequency scales, providing rich perceptual representations suitable for audio synthesis evaluation.

### Computational Characteristics

#### Forward Pass Performance
- **Forward JTFS**: ~5-10ms per sample
- **Jacobian computation**: ~5000ms per sample (**500× slower**)
- **Memory usage**: ~100MB per sample during differentiation

#### Primary Computational Bottleneck
The JTFS Jacobian computation dominates runtime due to:
- **Multi-scale complexity**: 13 temporal scales × 12 frequency filters
- **Second-order scattering**: Wavelets of wavelets create exponential path dependencies
- **Dense coupling**: Every output coefficient depends on all input parameters
- **Non-smooth operations**: Modulus `|·|` and log transforms complicate automatic differentiation

### Implementation Details

#### Forward Synthesis + Jacobian (`02_compute_pnp_jacobian.py`)

```python
# Define PNP forward operator: parameters → audio → JTFS coefficients
S_from_nu = taslp23.pnp_forward_factory(scaler, logscale, synth_type)

# Compute Jacobian using forward-mode AD (efficient for low-dim input)
dS_over_dnu = jacfwd(S_from_nu)

# For each parameter sample:
nu = torch.tensor(nus[i, :], requires_grad=True)
S = S_from_nu(nu)           # JTFS coefficients: [~20,762]
J = dS_over_dnu(nu)         # Jacobian: [20,762 × 5]

# Save to disk for training
np.save(S_path, S.detach().numpy())  # JTFS features
np.save(J_path, J.detach().numpy())  # Gradients
```

#### Riemannian Metrics (`12_compute_lmastep_fast.py`)

```python
# Load pre-computed Jacobians and compute M matrices
J = dS_over_dnu(nu_tensor)        # Shape: [20,762 × 5]
M = torch.matmul(J.T, J)          # Riemannian metric: [5 × 5]
sigma = torch.linalg.eigvals(M)   # Eigenvalue spectrum for weighting
```

#### Neural Network Training (`05_train_effnet_pnploss.py`)

```python
# Train EfficientNet for inverse synthesis: audio → parameters
model = cnn.EffNet(
    in_channels=1,
    outdim=5,              # Predict 5 physical parameters
    loss=loss_type,        # "weighted_p" uses PNP loss with M matrices
    LMA=LMA_config        # Levenberg-Marquardt acceleration
)

# Dataset loads pre-computed JTFS features and M matrices
dataset = cnn.DrumDataModule(
    feature="cqt",         # Can use CQT or JTFS features
    weight_dir=weight_dir, # Path to M matrices for loss weighting
    weight_type="novol"    # How to apply Riemannian weights
)
```

#### Optimization Strategies

- **Batch processing**: 200 samples/batch on MPS (MacBook Pro M3 Max)
- **Parallel I/O**: ThreadPoolExecutor with 8 workers for HDF5 storage
- **Mixed precision**: float32 on MPS, float64 on CPU for numerical stability
- **Pre-computation**: Jacobians computed offline due to 500× computational cost

### Research Applications

This JTFS-based pipeline enables:

- **Neural inverse synthesis**: Train networks to estimate physical parameters from audio using perceptual features
- **Perceptually-weighted losses**: Use Riemannian metrics (M matrices) to weight parameter estimation errors appropriately
- **Large-scale dataset generation**: Pre-compute JTFS features and gradients for 100,000+ audio samples efficiently
- **Parameter sensitivity analysis**: Understand how physical parameters affect perceptual characteristics through Jacobian analysis
- **Multi-modal training**: Compare different audio features (JTFS vs CQT) for parameter estimation tasks

### Alternative Approaches

Due to the computational expense, the research community often uses:
- **Simpler spectral losses** (FFT-based instead of JTFS)
- **Pre-computed representations** (current approach)
- **Approximate differentiation** methods
- **Cached intermediate results**

The **500× computational overhead** for Jacobian computation makes pre-computation essential for this pipeline. By computing JTFS features and gradients offline (`02_compute_pnp_jacobian.py`), the project enables practical neural network training on large datasets (100,000+ samples) while maintaining the rich perceptual information that JTFS provides. The resulting pre-computed features and Riemannian metrics enable efficient training of EfficientNet models for inverse synthesis tasks.
