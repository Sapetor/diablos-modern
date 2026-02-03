# Optimization Primitives

DiaBloS includes low-level building blocks for constructing optimization algorithms visually using feedback loops, rather than just configuring an external optimizer.

## Overview

These blocks allow you to **build optimization algorithms as block diagrams**. For example, gradient descent:

```
X_{k+1} = X_k - α * ∇f(X_k)
```

can be constructed by connecting blocks in a feedback loop where each simulation step corresponds to one optimization iteration.

## Key Concepts

- **Iteration = Time Step**: Each simulation step is one algorithm iteration
- **Modular Gradient**: Build finite difference structure with VectorPerturb + ObjectiveFunction blocks
- **Feedback Loop**: StateVariable holds state, VectorSum computes updates

---

## ObjectiveFunction

Evaluates f(x) from a Python expression.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `expression` | string | `"x1**2 + x2**2"` | Python expression using x1, x2, ... |
| `variables` | string | `"x1,x2"` | Comma-separated variable names |

### Inputs
- **Port 0**: Vector x = [x1, x2, ...]

### Outputs
- **Port 0**: Scalar f(x)

### Available Functions
`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pi`, `e`

### Example Expressions
- Quadratic: `x1**2 + x2**2`
- Rosenbrock: `(1-x1)**2 + 100*(x2-x1**2)**2`
- Custom: `sin(x1) + exp(-x2**2)`

---

## VectorPerturb

Perturbs x[index] by epsilon for finite difference gradient computation.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `index` | int | 0 | Which component to perturb (0-indexed) |
| `epsilon` | float | 1e-6 | Perturbation size |

### Inputs/Outputs
- **Input**: Vector x
- **Output**: Vector x with x[index] += epsilon

### Usage
Create one VectorPerturb block per dimension, each with a different index, to compute finite difference gradients.

---

## NumericalGradient

Computes gradient ∇f from finite difference inputs.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dimension` | int | 2 | Number of variables |
| `epsilon` | float | 1e-6 | Perturbation size (must match VectorPerturb) |
| `method` | string | "forward" | "forward" or "central" difference |

### Inputs
- **Port 0**: f(x) - function value at center point
- **Port 1..n**: f(x + ε·eᵢ) - perturbed values for each dimension
- For central: also f(x - ε·eᵢ) inputs

### Outputs
- **Port 0**: Gradient vector ∇f

### Formulas
- Forward: ∇f[i] = (f(x + ε·eᵢ) - f(x)) / ε
- Central: ∇f[i] = (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)

---

## StateVariable

Holds optimization state x(k) across iterations.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `initial_value` | list | [1.0, 1.0] | Starting value |
| `dimension` | int | 2 | Number of state variables |

### Inputs/Outputs
- **Input**: x_next - next state value
- **Output**: x_current - current state value

### Behavior
1. First iteration: outputs initial_value
2. Each subsequent iteration: outputs the x_next received in previous step

---

## VectorGain

Scales a vector by a scalar: y = α * x

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gain` | float | 1.0 | Scalar multiplier |

### Usage
Use negative gain for gradient descent: `gain = -0.1` gives y = -α * ∇f

---

## VectorSum

Adds or subtracts multiple vector inputs.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `signs` | string | "++" | String of '+' and '-' for each input |

### Examples
- `"++"`: y = x1 + x2
- `"+-"`: y = x1 - x2 (gradient descent update)
- `"++-"`: y = x1 + x2 - x3

---

## LinearSystemSolver

Solves the linear system Ax = b.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `method` | string | "direct" | "direct", "lstsq", or "pinv" |
| `dimension` | int | 2 | System dimension |
| `regularization` | float | 0.0 | Tikhonov regularization |

### Inputs
- **Port 0**: Matrix A (can be flattened)
- **Port 1**: Vector b

### Outputs
- **Port 0**: Solution vector x

---

## RootFinder

Computes one Newton step for solving F(x) = 0.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `expressions` | string | `"x1**2+x2-1, x1+x2**2-1"` | Comma-separated F expressions |
| `variables` | string | `"x1,x2"` | Variable names |
| `epsilon` | float | 1e-6 | Perturbation for numerical Jacobian |
| `damping` | float | 1.0 | Step damping factor (0 < damping ≤ 1) |

### Algorithm
Newton's method: x_{k+1} = x_k - J(x_k)^{-1} F(x_k)

The Jacobian is computed numerically using finite differences.

---

## ResidualNorm

Computes the norm of a vector for convergence monitoring.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `norm_type` | string | "2" | "1" (Manhattan), "2" (Euclidean), "inf" (max) |

### Usage
Monitor convergence by connecting to a Scope:
- ‖∇f‖ < tol → gradient descent converged
- ‖F(x)‖ < tol → root finding converged

---

## Momentum

Momentum-accelerated gradient descent.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `alpha` | float | 0.01 | Learning rate |
| `beta` | float | 0.9 | Momentum coefficient |

### Algorithm
```
v = β * v - α * ∇f
x_new = x + v
```

### Inputs/Outputs
- **Input**: Gradient ∇f
- **Output**: Update vector v (add to x)

The velocity state is maintained internally.

---

## Adam

Adam optimizer with adaptive learning rates.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `alpha` | float | 0.001 | Learning rate |
| `beta1` | float | 0.9 | First moment decay |
| `beta2` | float | 0.999 | Second moment decay |
| `epsilon` | float | 1e-8 | Numerical stability |

### Algorithm
```
m = β₁*m + (1-β₁)*∇f           # First moment
v = β₂*v + (1-β₂)*∇f²          # Second moment
m̂ = m / (1 - β₁ᵗ)              # Bias correction
v̂ = v / (1 - β₂ᵗ)
update = -α * m̂ / (√v̂ + ε)
```

### Inputs/Outputs
- **Input**: Gradient ∇f
- **Output**: Update vector (add to x)

---

## Example: Gradient Descent Diagram

```
                                    ┌──────────────────┐
                              ┌────▶│ ObjectiveFunction│──────┐
                              │     │ f(x) = x1²+x2²   │      │
                              │     └──────────────────┘      │
┌─────────────────┐           │                               │
│  StateVariable  │───────────┼──▶[VectorPerturb(0)]──▶[Obj]──┼──▶┌───────────────┐
│  x₀=[5,5]       │           │                               │   │NumericalGrad  │
└────────▲────────┘           └──▶[VectorPerturb(1)]──▶[Obj]──┼──▶│ dim=2, ε=1e-6 │
         │                                                    │   └───────┬───────┘
         │                    ┌────────────────────┐          │           │
         │                    │      Scope         │◀─────────┘           │
         │                    │  (monitor f value) │                      ▼
         │                    └────────────────────┘              ┌───────────┐
         │                                                        │ VectorGain│
         │              ┌─────────────┐                           │  α=-0.1   │
         └──────────────│  VectorSum  │◀──────────────────────────┴───────────┘
                        │  x + update │
                        └─────────────┘
```

## Example Diagrams

| File | Description |
|------|-------------|
| `gradient_descent_demo.diablos` | Basic gradient descent on f(x) = x₁² + x₂² |
| `momentum_demo.diablos` | Momentum optimizer on Rosenbrock function |
| `adam_demo.diablos` | Adam optimizer on Rosenbrock function |
| `newton_method_demo.diablos` | Newton's method for nonlinear system |
| `linear_system_demo.diablos` | Solving Ax = b with verification |

---

## Tips

1. **Set sim_dt = 1.0** - each step is one iteration
2. **Match epsilon** between VectorPerturb and NumericalGradient
3. **Use ResidualNorm + Scope** to monitor convergence
4. **Start with small learning rates** for nonlinear problems
5. **Adam is more robust** than plain gradient descent for difficult problems
