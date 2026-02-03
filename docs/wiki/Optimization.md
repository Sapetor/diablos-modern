# Optimization Blocks

DiaBloS supports parameter optimization through a set of specialized blocks that work together to define and solve optimization problems.

## Overview

The optimization workflow consists of:
1. **Parameter** blocks define variables to optimize
2. **CostFunction** block evaluates the objective to minimize
3. **Constraint** blocks (optional) define inequality constraints
4. **Optimizer** block configures and triggers optimization

## How to Run Optimization

1. Load or create a diagram with optimization blocks
2. Run the simulation normally first to verify it works
3. Click **Tools → Run Optimization** (or press `Ctrl+Shift+O`)
4. The optimizer will iterate, running simulations with different parameter values
5. Results are displayed in the console and parameter blocks are updated

---

## Parameter Block

Defines a parameter that can be optimized.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | string | "param" | Parameter name (used as workspace variable) |
| `value` | float | 1.0 | Initial/current value |
| `lower` | float | 0.0 | Lower bound for optimization |
| `upper` | float | 10.0 | Upper bound for optimization |
| `scale` | string | "linear" | Scale: "linear" or "log" |
| `fixed` | bool | False | If True, parameter is not optimized |

### Outputs
- **Port 0**: Current parameter value

### Example Usage
Connect the Parameter output to a Gain block to make the gain optimizable:
```
[Parameter: Kp] → [Gain: Controller]
```

---

## CostFunction Block

Computes the optimization objective (cost) to minimize.

### Cost Types
| Type | Formula | Best For |
|------|---------|----------|
| `ISE` | ∫ e(t)² dt | Integrated Squared Error - penalizes large errors |
| `IAE` | ∫ |e(t)| dt | Integrated Absolute Error - equal weight to all errors |
| `ITAE` | ∫ t·|e(t)| dt | Time-weighted IAE - penalizes long settling time |
| `ITSE` | ∫ t·e(t)² dt | Time-weighted ISE |
| `SSE` | e(t_final)² | Steady-State Error only |
| `terminal` | |e(t_final)| | Final value only |

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cost_type` | string | "ISE" | Cost function type |
| `weight` | float | 1.0 | Weight multiplier |
| `t_start` | float | 0.0 | Start time for integration |
| `t_end` | float | -1.0 | End time (-1 = simulation end) |

### Inputs
- **Port 0**: Error signal e(t) to evaluate

---

## Optimizer Block

Meta-block that configures the optimization algorithm.

### Optimization Methods
| Method | Description | Supports Constraints |
|--------|-------------|---------------------|
| `L-BFGS-B` | Bounded quasi-Newton (default) | Bounds only |
| `SLSQP` | Sequential Least Squares | Yes |
| `Nelder-Mead` | Derivative-free simplex | No |
| `differential_evolution` | Global search | Bounds only |
| `Powell` | Conjugate direction | No |
| `COBYLA` | Linear approximation | Yes |

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `method` | string | "L-BFGS-B" | Optimization algorithm |
| `max_iter` | int | 100 | Maximum iterations |
| `tol` | float | 1e-6 | Convergence tolerance |
| `use_constraints` | bool | True | Enable constraint handling |
| `verbose` | bool | True | Print progress to console |

---

## Constraint Block

Defines an inequality constraint g(x) ≤ limit.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `type` | string | "max" | "max" (≤ limit) or "min" (≥ limit) |
| `limit` | float | 1.0 | Constraint threshold |
| `name` | string | "constraint" | Constraint name for reporting |

### Inputs
- **Port 0**: Signal to constrain

### Example
To limit overshoot to 20%:
```
[System Output] → [Constraint: max=1.2]
```

---

## DataFit Block

Special cost function for fitting model to experimental data.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `data_file` | string | "" | Path to CSV file with time,value columns |
| `error_metric` | string | "MSE" | "MSE", "RMSE", or "MAE" |
| `interpolation` | string | "linear" | "linear" or "nearest" |

### Inputs
- **Port 0**: Model output signal to compare with data

---

## Example Diagrams

### 1. Basic Gain Optimization (`optimization_basic_demo.diablos`)
Optimizes a proportional gain Kp to minimize ISE for step response.

### 2. PID Tuning (`optimization_pid_tuning_demo.diablos`)
Auto-tunes Kp, Ki, Kd parameters using ITAE cost function.

### 3. Constrained Optimization (`optimization_constrained_demo.diablos`)
Optimizes with overshoot constraint using SLSQP method.

### 4. Data Fitting (`optimization_data_fit_demo.diablos`)
Calibrates model parameters to match experimental CSV data.

---

## Tips

1. **Start with good initial values** - optimization finds local minima
2. **Set reasonable bounds** - too wide makes search inefficient
3. **Use `log` scale for gains** that span orders of magnitude
4. **SLSQP** is required for constraint blocks to work
5. **Check verbose output** for convergence info
