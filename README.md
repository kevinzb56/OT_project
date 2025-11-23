# Optimizing Deep Neural Network Weights using Fast Particle Swarm Optimization (PSO)

## Overview

This project explores the viability of **Non-Gradient Optimization** for Deep Feedforward Neural Networks (DFNNs). Specifically, it implements a **Speed-Optimized Particle Swarm Optimization (Fast PSO)** algorithm to train neural networks, replacing standard Backpropagation (Gradient Descent).

While gradient-based methods (Adam, SGD) are the industry standard, they often get trapped in local minima. Meta-heuristics like PSO offer robust global search capabilities but are notoriously slow. This project aims to bridge that gap by implementing aggressive speed optimizations (Vectorization + JIT Compilation) to make PSO competitive with modern Gradient Descent.

### Research Goal

To determine if an aggressively optimized PSO algorithm can achieve classification accuracy comparable to or better than Gradient Descent while maintaining acceptable computational speed on complex classification tasks.

-----

## The Algorithm

The project implements a **Constriction-Factor Particle Swarm Optimization**. In this framework, every "particle" in the swarm represents a single candidate neural network.

### 1\. Particle Representation

The entire set of weights and biases for the neural network is flattened into a single 1D vector $\theta$. This vector serves as the **position** ($x$) of a particle in the high-dimensional search space.

  * **Dimension ($D$):** Total number of weights + biases in the network (e.g., \~1500 dimensions for a 30-48-1 network).
  * **Population ($N$):** 10-15 particles.

### 2\. Update Equations

At every iteration $t$, each particle $i$ updates its velocity ($v$) and position ($x$) based on its own experience (Personal Best) and the swarm's experience (Global Best).

**Velocity Update:**
$$v_{i}^{t+1} = w \cdot v_{i}^{t} + c_1 \cdot r_1 \cdot (pbest_i - x_{i}^{t}) + c_2 \cdot r_2 \cdot (gbest - x_{i}^{t})$$

**Position Update:**
$$x_{i}^{t+1} = x_{i}^{t} + v_{i}^{t+1}$$

Where:

  * $w$: Inertia weight (controls exploration vs exploitation).
  * $c_1, c_2$: Cognitive and Social coefficients.
  * $r_1, r_2$: Random vectors sampled from a uniform distribution $[0, 1]$.
  * $pbest_i$: The best position found by particle $i$ so far.
  * $gbest$: The best position found by the entire swarm so far.

### 3\. Hyperparameters & Constraints

To ensure stability and convergence, the following specific parameters were tuned:

  * **Inertia ($w$):** `0.7298`
  * **Coefficients ($c_1, c_2$):** `1.49618`
  * **Velocity Clamping:** Velocities are clipped to $[-1.0, 1.0]$ to prevent explosion.
  * **Position Clipping:** Weights are clipped to $[-5.0, 5.0]$ to ensure valid network parameters.
  * **Initialization:** Particles are initialized using **He/Xavier** scaling ($\sqrt{2/D}$) rather than purely random values, starting the search closer to viable solutions.

### 4\. Fitness Function

The "fitness" to minimize is the **Binary Cross-Entropy Loss** of the neural network over the training batch:
$$Fitness(\theta) = -\frac{1}{N_{samples}} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

-----

## Key Features

  * **Hybrid & Fast PSO Implementation**: A custom PSO built from scratch, specifically tuned for high-dimensional weight optimization.
  * **Numba JIT Acceleration**: Activation functions (ReLU, Sigmoid) and critical loops are compiled using Numba's Just-In-Time compiler for C-like performance.
  * **Full Vectorization**: Core PSO update logic utilizes NumPy vectorization to eliminate slow Python loops.
  * **Early Stopping**: The algorithm monitors the global best fitness and terminates if no improvement is seen for 15 iterations, saving computation time.
  * **Benchmarking Suite**: Automated comparison tools against Scikit-Learn’s `MLPClassifier` (Adam solver).

-----

## Performance Results

The implementation was tested on a Synthetic Dataset and the Wisconsin Breast Cancer Dataset (WBCD).

| Dataset | Metric | Fast PSO | Gradient Descent (Adam) | Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Small Synthetic** | Test Accuracy | **85.00%** | 77.50% | +7.50%|
| | Training Time | 0.56s | **0.09s** | PSO is 6x slower |
| **Wisconsin Cancer**| Test Accuracy | 96.49% | **98.25%** | -1.76% |
| | Training Time | 0.22s | **0.13s** | PSO is 1.6x slower |

**Conclusion:** While PSO remains slower than the highly optimized C-backend of Scikit-Learn, the **Fast PSO** implementation proves it can achieve superior generalization on noisy/synthetic data and highly competitive accuracy on real-world medical data.

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/fast-pso-neural-net.git
    cd fast-pso-neural-net
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install numpy matplotlib scikit-learn seaborn numba
    ```

-----

## Usage

The project is contained within a Jupyter Notebook. You can run the full experiment suite directly.

1.  **Open the Notebook:**

    ```bash
    jupyter notebook NeuralNetsUsingPSO.ipynb
    ```

2.  **Run the Experiments:**
    Execute the cells to trigger the `run_small_test()` and `run_large_test()` functions.

### Customizing the Network

You can use the `FastNeuralNetwork` and `FastPSO` classes in your own scripts:

```python
from your_script import FastNeuralNetwork, FastPSO

# 1. Define Network Architecture
input_dim = 30
hidden_layers = [48, 24]
output_dim = 1
nn = FastNeuralNetwork(input_dim, hidden_layers, output_dim)

# 2. Define Fitness Function (Loss)
def fitness_func(weight_vector):
    weights = nn.vector_to_weights(weight_vector)
    return nn.compute_loss_batch(X_train, y_train, weights)

# 3. Run Optimization
total_weights = nn.get_weight_vector_size()
pso = FastPSO(dim=total_weights, population_size=15, max_iter=100)
best_weights_vector, best_loss = pso.optimize(fitness_func)
```

-----

## Contributors

  * **[Viraj Vora](https://github.com/viraj200524)** - *Veermata Jijabai Technological Institute (VJTI)*
  * **[Kevin Shah](https://github.com/kevinzb56)** - *Veermata Jijabai Technological Institute (VJTI)*

-----

## References

  * [Optimizing Neural Network Weights Using Nature-Inspired Algorithms](https://arxiv.org/pdf/2105.09983)
  * [UCI Machine Learning Repository: Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
