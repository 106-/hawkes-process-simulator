"""
Hawkes Process Simulation Module

Implements univariate Hawkes process with various kernel functions:
λ(t) = μ + Σ(i: t_i < t) φ(t - t_i)

where φ(t) is the kernel function (exponential, power-law, etc.)
"""

import numpy as np
from typing import Optional
from src.kernels import Kernel, ExponentialKernel


class HawkesProcess:
    """
    Univariate Hawkes Process with customizable kernel.

    Parameters:
        mu (float): Baseline intensity (background rate)
        kernel (Kernel): Kernel function (default: exponential)
        alpha (float, optional): For backward compatibility with exponential kernel
        beta (float, optional): For backward compatibility with exponential kernel

    Usage:
        # New style (with kernel object)
        kernel = ExponentialKernel(alpha=0.5, beta=1.0)
        hp = HawkesProcess(mu=1.0, kernel=kernel)

        # Old style (backward compatible)
        hp = HawkesProcess(mu=1.0, alpha=0.5, beta=1.0)
    """

    def __init__(self, mu: float, alpha: float = None, beta: float = None,
                 kernel: Optional[Kernel] = None):
        """
        Initialize Hawkes process with parameters.

        Args:
            mu: Baseline intensity
            alpha: Excitation parameter (for exponential kernel)
            beta: Decay rate (for exponential kernel)
            kernel: Kernel object (if provided, alpha/beta are ignored)
        """
        self.mu = mu

        # Backward compatibility: if alpha and beta provided, create exponential kernel
        if kernel is None:
            if alpha is not None and beta is not None:
                kernel = ExponentialKernel(alpha, beta)
                self.alpha = alpha
                self.beta = beta
            else:
                raise ValueError("Either provide kernel, or both alpha and beta")
        else:
            # Extract alpha/beta if it's an exponential kernel
            if isinstance(kernel, ExponentialKernel):
                self.alpha = kernel.alpha
                self.beta = kernel.beta
            else:
                self.alpha = None
                self.beta = None

        self.kernel = kernel

    def check_stability(self) -> bool:
        """
        Check if the process is stable.

        Returns:
            True if stable, False otherwise
        """
        return self.kernel.check_stability()

    def compute_intensity(self, t: float, event_times: np.ndarray) -> float:
        """
        Compute intensity λ(t) at time t given event history.

        λ(t) = μ + Σ(i: t_i < t) φ(t - t_i)

        Args:
            t: Time point at which to compute intensity
            event_times: Array of previous event times

        Returns:
            Intensity value at time t
        """
        # Start with baseline intensity
        intensity = self.mu

        # Add contributions from past events using kernel
        if len(event_times) > 0:
            past_events = event_times[event_times < t]
            if len(past_events) > 0:
                dt = t - past_events
                contributions = self.kernel.evaluate(dt)
                intensity += np.sum(contributions)

        return intensity

    def simulate(self, T_max: float, seed: int = None, max_events: int = 10000,
                 max_iterations: int = 100000) -> np.ndarray:
        """
        Simulate Hawkes process using Ogata's thinning algorithm.

        Algorithm:
        1. Start with empty event list and t=0
        2. Compute current intensity λ(t)
        3. Find upper bound λ* ≥ λ(t)
        4. Sample inter-event time from Exp(λ*)
        5. Accept/reject with probability λ(t)/λ*
        6. Repeat until t > T_max

        Args:
            T_max: Maximum simulation time
            seed: Random seed for reproducibility
            max_events: Maximum number of events (safety limit)
            max_iterations: Maximum loop iterations (prevents infinite loops)

        Returns:
            Array of event times
        """
        if seed is not None:
            np.random.seed(seed)

        events = []
        t = 0.0
        iterations = 0

        # Get kernel upper bound
        kernel_upper_bound = self.kernel.get_upper_bound()

        while t < T_max and len(events) < max_events and iterations < max_iterations:
            iterations += 1
            # Current intensity
            lambda_t = self.compute_intensity(t, np.array(events))

            # Upper bound (intensity can jump by at most kernel upper bound)
            lambda_star = lambda_t + kernel_upper_bound

            # Sample exponential waiting time
            if lambda_star > 0:
                s = np.random.exponential(1.0 / lambda_star)
            else:
                break

            t = t + s

            if t > T_max:
                break

            # Compute intensity at new time
            lambda_new = self.compute_intensity(t, np.array(events))

            # Accept/reject
            u = np.random.uniform(0, 1)
            if u <= lambda_new / lambda_star:
                events.append(t)

        return np.array(events)

    def get_intensity_trace(self, event_times: np.ndarray, T_max: float,
                           n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth intensity function for visualization.

        Args:
            event_times: Array of event times from simulation
            T_max: Maximum time
            n_points: Number of points for time grid

        Returns:
            Tuple of (time_grid, intensity_values)
        """
        time_grid = np.linspace(0, T_max, n_points)
        intensity = np.zeros(n_points)

        for i, t in enumerate(time_grid):
            intensity[i] = self.compute_intensity(t, event_times)

        return time_grid, intensity

    def get_branching_ratio(self) -> float:
        """
        Get the branching ratio.

        This represents the expected number of offspring per event.
        For exponential kernel: n = α/β
        For general kernel: n = ∫₀^∞ φ(t) dt

        Must be < 1 for stability.

        Returns:
            Branching ratio
        """
        # Try to use kernel-specific method first
        if hasattr(self.kernel, 'get_branching_ratio'):
            return self.kernel.get_branching_ratio()

        # Otherwise use integral
        return self.kernel.get_integral()

    def get_kernel_name(self) -> str:
        """Get the name of the kernel being used."""
        return self.kernel.get_name()

    def get_kernel_params(self) -> dict:
        """Get the parameters of the kernel."""
        return self.kernel.get_params_dict()
