# Example Diagrams

DiaBloS includes example diagrams demonstrating various features. Load examples via **File → Open** and navigate to the `examples/` folder.

## Running Examples

1. **Open** the example file (`.diablos`)
2. **Run** simulation (`F5` or Play button)
3. **View** results in Scope/FieldScope plots
4. For optimization examples: **Tools → Run Optimization**

---

## Optimization Examples

### optimization_basic_demo.diablos
**Basic Gain Tuning**

Optimizes a proportional controller gain Kp to minimize step response error.

- **System**: First-order plant with gain controller
- **Cost**: ISE (Integrated Squared Error)
- **Parameters**: Kp ∈ [0.1, 20]
- **Expected result**: Kp ≈ 5-10 depending on plant dynamics

### optimization_pid_tuning_demo.diablos
**PID Auto-Tuning**

Automatically tunes all three PID parameters.

- **System**: Second-order plant with PID controller
- **Cost**: ITAE (penalizes slow settling)
- **Parameters**: Kp, Ki, Kd with bounds
- **Method**: L-BFGS-B (default)

### optimization_constrained_demo.diablos
**Constrained Optimization**

Optimizes with overshoot constraint.

- **System**: Underdamped plant
- **Cost**: ISE
- **Constraint**: Output ≤ 1.2 (20% max overshoot)
- **Method**: SLSQP (required for constraints)

### optimization_data_fit_demo.diablos
**Model Calibration**

Fits model parameters to match experimental data from CSV file.

- **Data**: `examples/sample_data.csv`
- **Cost**: MSE between model and data
- **Use case**: System identification, parameter estimation

---

## PDE Verification Examples

These examples compare numerical solutions against known analytical solutions.

### heat_equation_1d_verification.diablos
**1D Heat Diffusion**

- **Equation**: ∂T/∂t = α·∂²T/∂x²
- **Initial**: T(x,0) = sin(πx/L)
- **Analytical**: T(x,t) = sin(πx/L)·exp(-απ²t/L²)
- **Decay rate**: λ = α(π/L)² ≈ 0.987 for α=0.1, L=1

### wave_equation_1d_verification.diablos
**1D Standing Wave**

- **Equation**: ∂²u/∂t² = c²·∂²u/∂x²
- **Initial**: u(x,0) = sin(πx/L), ∂u/∂t = 0
- **Analytical**: u(x,t) = sin(πx/L)·cos(πct/L)
- **Period**: T = 2L/c = 2.0s for c=1, L=1

### advection_equation_1d_verification.diablos
**1D Traveling Wave**

- **Equation**: ∂c/∂t + v·∂c/∂x = 0
- **Initial**: Gaussian pulse at x = L/4
- **Analytical**: Pulse travels at velocity v
- **Note**: First-order upwind has numerical diffusion

### diffusion_reaction_1d_verification.diablos
**Diffusion-Reaction (Linear)**

- **Equation**: ∂c/∂t = D·∂²c/∂x² - k·c
- **Initial**: c(x,0) = sin(πx/L)
- **Analytical**: c(x,t) = sin(πx/L)·exp(-(Dπ²/L² + k)t)
- **Decay rate**: λ = Dπ²/L² + k ≈ 1.487

### heat_equation_2d_verification.diablos
**2D Heat Diffusion**

- **Equation**: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
- **Initial**: T = sin(πx)·sin(πy)
- **Analytical**: T(x,y,t) = sin(πx)·sin(πy)·exp(-2απ²t)
- **Uses**: Interactive time slider

---

## Optimization Primitives Examples

These examples demonstrate building optimization algorithms visually using feedback loops.

### gradient_descent_demo.diablos
**Gradient Descent on Quadratic**

- **Objective**: f(x) = x₁² + x₂²
- **Method**: Finite difference gradient with learning rate α = 0.1
- **Initial**: x₀ = [5, 5]
- **Expected**: Converges to [0, 0] in ~50 iterations

### momentum_demo.diablos
**Momentum Optimizer on Rosenbrock**

- **Objective**: f(x) = (1-x₁)² + 100(x₂-x₁²)²
- **Method**: Momentum with β = 0.9, α = 0.0001
- **Initial**: x₀ = [5, 5]
- **Minimum**: [1, 1]

### adam_demo.diablos
**Adam Optimizer on Rosenbrock**

- **Objective**: Rosenbrock function
- **Method**: Adam with default hyperparameters
- **Initial**: x₀ = [-2, 2]
- **Note**: More robust than basic gradient descent

### newton_method_demo.diablos
**Newton's Method for Nonlinear System**

- **System**: x₁² + x₂ - 1 = 0, x₁ + x₂² - 1 = 0
- **Method**: Newton-Raphson with numerical Jacobian
- **Solutions**: (0, 1) or (1, 0)
- **Convergence**: Quadratic near solution

### linear_system_demo.diablos
**Linear System Solver**

- **System**: Ax = b where A = [[2,1],[1,3]], b = [5, 7]
- **Solution**: x = [1.6, 1.8]
- **Verification**: Computes A*x to verify equals b

---

## Optimization Primitives Verification Examples

These examples compare numerical optimization results against known analytical solutions.

### gradient_descent_verification.diablos
**Gradient Descent Convergence Verification**

- **Objective**: f(x) = x₁² + x₂²
- **Initial**: x₀ = [5, 5], learning rate α = 0.1
- **Analytical**: x(k) = (1-2α)^k · x₀ = 0.8^k · [5, 5]
- **Comparison**: Side-by-side f(x) numerical vs analytical
- **Error metric**: |f_numerical - f_analytical| should be < 1e-10

### newton_method_verification.diablos
**Newton's Method Convergence Verification**

- **System**: x₁² + x₂ - 1 = 0, x₁ + x₂² - 1 = 0
- **Initial**: x₀ = [0.5, 0.5]
- **Analytical solution**: x* = [1, 0]
- **Verification**: Tracks ||F(x)|| and ||x - x*||
- **Expected**: Quadratic convergence, error < 1e-10

### linear_system_verification.diablos
**Linear System Solver Verification**

- **System**: Ax = b where A = [[2,1],[1,3]], b = [5, 7]
- **Analytical**: x* = [1.6, 1.8] (computed via A⁻¹b)
- **Verification**: Computes ||x - x*|| and ||Ax - b||
- **Expected**: Both error metrics < 1e-14

---

## Teaching Examples

These examples are designed for teaching optimization concepts with clear visualizations.

### learning_rate_comparison.diablos
**Effect of Learning Rate on Convergence**

Demonstrates four gradient descent runs on f(x) = x² with different learning rates:
- **α = 0.1**: Slow but stable convergence (factor 0.8)
- **α = 0.4**: Fast convergence near optimal (factor 0.2)
- **α = 0.6**: Oscillating convergence (factor -0.2)
- **α = 1.2**: Divergence (factor -1.4, |factor| > 1)

Key insight: Stability requires |1-2α| < 1, optimal is α = 0.5 for quadratic.

### optimizer_comparison.diablos
**Gradient Descent vs Momentum vs Adam**

Compares three optimization methods on the Rosenbrock function f(x) = (1-x₁)² + 100(x₂-x₁²)²:
- **GD**: Simple but struggles in narrow valleys
- **Momentum**: Builds velocity to navigate valleys faster
- **Adam**: Adaptive learning rates handle different curvatures

All start from [-1, 1] targeting minimum at [1, 1].

### convergence_rates.diablos
**Linear vs Quadratic Convergence**

Compares gradient descent (linear convergence) vs Newton's method on finding root of f(x) = 4x³:
- **Linear**: Error decreases by constant factor each iteration
- **Quadratic**: Error squares each iteration (much faster)
- **Log plot**: Linear convergence shows straight line, quadratic shows rapid drop

Visualizes why Newton converges in fewer iterations despite higher cost per step.

---

## Other Demo Examples

### pde_comparison_demo.diablos
Side-by-side comparison of Heat, Wave, and Advection equations.

### pde_neumann_bc_demo.diablos
Demonstrates insulated boundaries (Neumann BC) with energy conservation.

---

## Creating Your Own Examples

1. Build your diagram in the canvas
2. Set simulation parameters (time, dt) in Settings
3. Add Scope/FieldScope blocks to visualize results
4. Save as `.diablos` file
5. Add `_verification_notes` JSON key for documentation (optional)

### Example JSON structure:
```json
{
    "sim_data": { "sim_time": 10.0, "sim_dt": 0.01 },
    "blocks_data": [ ... ],
    "lines_data": [ ... ],
    "_verification_notes": {
        "problem": "Description of the problem",
        "analytical_solution": "Formula",
        "parameters": { "key": "value" }
    }
}
```
