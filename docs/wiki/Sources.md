# Sources Blocks

List of available blocks in the **Sources** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Chirp](#chirp) | Swept-frequency cosine (chirp) source. |
| [Constant](#constant) | Outputs a constant value. |
| [FromFile](#fromfile) | Replays a time-series loaded from a CSV/NPZ/MAT/text file. |
| [Impulse](#impulse) | Discrete impulse (Dirac delta approximation). |
| [Noise](#noise) | White Noise Generator. |
| [PRBS](#prbs) | Pseudo-Random Binary Sequence (PRBS). |
| [Ramp](#ramp) | Generates a Linear Ramp signal. |
| [RandomSource](#randomsource) | Seeded random value, held between sample instants. |
| [Sine](#sine) | Generates a Sinusoidal signal. |
| [Step](#step) | Generates a Step function. |
| [WaveGenerator](#wavegenerator) | Generates periodic waveforms (Sine, Square, Triangle, Sawtooth). |

---

### Constant

Outputs a constant value.

Parameters:
- Value: The constant output value (scalar or vector).

Usage:
Useful for setpoints, constant parameters, or enabling blocks.
To create a vector, use [v1, v2, ...].

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | float | `1.0` | Constant output value. |

**Ports**: 0 In, 1 Out

---

### Noise

White Noise Generator.

Generates random numbers with a Normal (Gaussian) distribution.

Parameters:
- Mean: Average value (center).
- Std Dev: Standard Deviation (spread).
- Seed: Random seed for reproducibility (0 = random).

Usage:
Simulate sensor noise or process disturbances.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mu` | float | `0.0` | The mean of the noise. |
| `sigma` | float | `1.0` | The standard deviation of the noise. |

**Ports**: 0 In, 1 Out

---

### PRBS

Pseudo-Random Binary Sequence (PRBS).

Generates a binary signal (-Amp, +Amp) that approximates white noise.
Useful for System Identification.

Parameters:
- Amplitude: Height of the binary steps.
- Clock Period: Time duration of each step.

Usage:
Apply to system input to estimate frequency response (rich frequency content).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `high` | float | `1.0` | Value for logic high. |
| `low` | float | `0.0` | Value for logic low. |
| `bit_time` | float | `0.1` | Seconds each bit is held. |
| `order` | int | `7` | LFSR order (sequence length 2^order-1). |
| `seed` | int | `1` | Non‑zero initial LFSR state. |

**Ports**: 0 In, 1 Out

---

### Ramp

Generates a Linear Ramp signal.

Output increases linearly with time: y = Slope * (t - Start Time).
Output is 0 before Start Time.

Parameters:
- Slope: Rate of change (dy/dt).
- Start Time: Time (seconds) when the ramp starts.

Usage:
Used to test tracking performance or generate sweeping signals.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `slope` | float | `1.0` | The slope of the ramp. |
| `delay` | float | `0.0` | The delay of the ramp. |

**Ports**: 0 In, 1 Out

---

### Sine

Generates a Sinusoidal signal.

y(t) = Amplitude * sin(Frequency * t + Phase) + Bias

Parameters:
- Amplitude: Peak value.
- Frequency: Angular frequency (rad/s).
- Phase: Initial phase shift (rad).
- Bias: DC offset added to the signal.

Usage:
Standard test signal for frequency response analysis.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `amplitude` | float | `1.0` | The amplitude of the sine wave. |
| `omega` | float | `1.0` | The angular frequency of the sine wave. |
| `init_angle` | float | `0.0` | The initial angle of the sine wave. |

**Ports**: 0 In, 1 Out

---

### Step

Generates a Step function.

Output is 0 before 'Delay' time, and 'Final Value' afterwards.

Parameters:
- Final Value: The height of the step.
- Step Time: Time (seconds) when the step occurs.

Usage:
Commonly used to test step response of control systems.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | float | `1.0` | The value of the step. |
| `delay` | float | `0.0` | The delay of the step. |
| `type` | string | `up` | up, down, pulse, constant |
| `pulse_start_up` | bool | `True` | If type is pulse, defines if it starts up or down. |

**Ports**: 0 In, 1 Out

---

### WaveGenerator

Generates various periodic waveforms.

**Output Equation**:
$y(t) = Bias + Amplitude \times Waveform(2\pi \cdot Frequency \cdot t + Phase)$

**Waveforms**:
- **Sine**: Standard sinusoidal.
- **Square**: Switch between -1 and +1.
- **Triangle**: Linear ramps up and down (50% duty).
- **Sawtooth**: Linear ramp up, instant reset.

**Usage**:
Versatile signal source for testing system response to different excitations.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `waveform` | choice | `sine` | Shape of the wave. |
| `amplitude` | float | `1.0` | Peak amplitude (from zero). |
| `frequency` | float | `1.0` | Frequency in Hz. |
| `phase` | float | `0.0` | Phase shift in radians. |
| `bias` | float | `0.0` | Vertical offset (DC component). |

**Ports**: 0 In, 1 Out

---
### Chirp

Swept-frequency cosine (chirp) source.

The instantaneous frequency sweeps from `f0` at `t = 0` to `f1` at `t = t1`
following the chosen `method`. The block is a pure function of time, so it
carries no state between steps.

Usage:
Frequency-response identification and sweeping a plant through its resonances.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `f0` | float | `0.0` | Start frequency in Hz (at t=0). |
| `f1` | float | `10.0` | End frequency in Hz (at t=t1). |
| `t1` | float | `10.0` | Time at which the frequency reaches f1 (s). |
| `amplitude` | float | `1.0` | Peak amplitude of the signal. |
| `method` | choice | `linear` | Frequency sweep profile: `linear`, `logarithmic`, `quadratic`. |

**Ports**: 0 In, 1 Out

---

### FromFile

Replays a recorded time-series from a file as a driving signal.

Reads `(time, signal)` columns from a CSV / NPZ / MAT / text file and outputs the
value interpolated to the current simulation time.

Usage:
Drive a model with measured data, or replay a signal exported from a previous
run (`Export` block, or the headless `run` CSV).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `data_file` | string | `` | Path to the data file (.csv/.npz/.mat/.txt). |
| `time_col` | string | `t` | Time column: name or 0-based numeric index. |
| `signal_col` | string | `y` | Signal column: name or 0-based numeric index. |
| `interpolation` | choice | `linear` | Interpolation: `linear`, `zoh` (step), or `nearest`. |
| `end_behavior` | choice | `hold` | Past the last sample: `hold` the last value or `loop`. |

**Ports**: 0 In, 1 Out

---

### Impulse

Discrete impulse (Dirac delta approximation).

Outputs `value/dt` for one time step at the `delay` time and 0 elsewhere, so the
integral of the output equals `value`.

Usage:
Impulse responses of transfer functions and state-space systems.

!!! note
    The `Impulse` block runs on the interpreter path. A `Step` block set to
    *impulse* mode is likewise rejected by the Python-script exporter.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | float | `1.0` | Impulse strength (area). |
| `delay` | float | `0.0` | Time when the impulse fires. |

**Ports**: 0 In, 1 Out

---

### RandomSource

Random value drawn from a seeded RNG and held between sample instants.

Distributions: `uniform` (U[low, high]), `bernoulli` (1.0 with probability `p`),
`normal` (N(mu, sigma)) and `randint` (integer in [low, high], inclusive).

Usage:
Gate a `Switch` control port for packet dropping, or feed the `tau` port of a
[VariableTransportDelay](Control.md#variabletransportdelay) to model random
latency. Because the block exposes a `seed`, it takes a derived sub-seed in a
[Monte Carlo ensemble](../user-guide/analysis.md#monte-carlo-ensembles), so a
whole experiment is reproducible from one master seed.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `distribution` | choice | `uniform` | Distribution: `uniform`, `bernoulli`, `normal`, `randint`. |
| `p` | float | `0.5` | Bernoulli success probability (0..1). |
| `low` | float | `0.0` | Lower bound for uniform / randint. |
| `high` | float | `1.0` | Upper bound for uniform / randint. |
| `mu` | float | `0.0` | Mean for the normal distribution. |
| `sigma` | float | `1.0` | Standard deviation for the normal distribution. |
| `sample_time` | float | `0.0` | Sample period (s). 0 = every step (use dtime). |
| `seed` | int | `0` | RNG seed (0 = non-reproducible, nonzero = reproducible). |

**Ports**: 0 In, 1 Out

---
