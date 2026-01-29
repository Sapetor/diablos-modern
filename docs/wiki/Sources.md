# Sources Blocks

List of available blocks in the **Sources** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Constant](#constant) | Outputs a constant value. |
| [Noise](#noise) | White Noise Generator. |
| [PRBS](#prbs) | Pseudo-Random Binary Sequence (PRBS). |
| [Ramp](#ramp) | Generates a Linear Ramp signal. |
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
