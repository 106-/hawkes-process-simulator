"""
Kernel Functions for Hawkes Processes

Implements various kernel (excitation) functions for Hawkes processes.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf


class Kernel(ABC):
    """カーネル関数の抽象基底クラス"""

    @abstractmethod
    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        時間 t での寄与を計算

        Args:
            t: 経過時間（スカラーまたは配列）

        Returns:
            カーネルの値
        """
        pass

    @abstractmethod
    def get_upper_bound(self) -> float:
        """
        カーネルの上界を返す（Ogataアルゴリズム用）

        Returns:
            カーネルの最大値
        """
        pass

    @abstractmethod
    def check_stability(self) -> bool:
        """
        安定条件をチェック

        Returns:
            True if stable, False otherwise
        """
        pass

    @abstractmethod
    def get_integral(self, T: float = np.inf) -> float:
        """
        カーネルの積分値を計算（0からTまで）

        Args:
            T: 積分の上限

        Returns:
            積分値
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """カーネルの名前を返す"""
        pass

    @abstractmethod
    def get_params_dict(self) -> dict:
        """パラメータの辞書を返す"""
        pass


class ExponentialKernel(Kernel):
    """
    指数減衰カーネル: α * exp(-β * t)

    最も一般的なカーネル。短時間記憶を持つ。
    """

    def __init__(self, alpha: float, beta: float):
        """
        Args:
            alpha: 励起の強さ
            beta: 減衰率
        """
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t)
        result = np.where(t >= 0, self.alpha * np.exp(-self.beta * t), 0.0)
        return float(result) if result.ndim == 0 else result

    def get_upper_bound(self) -> float:
        return self.alpha

    def check_stability(self) -> bool:
        # 分岐比 n = α/β < 1
        return self.alpha < self.beta

    def get_integral(self, T: float = np.inf) -> float:
        if T == np.inf:
            return self.alpha / self.beta
        return (self.alpha / self.beta) * (1 - np.exp(-self.beta * T))

    def get_branching_ratio(self) -> float:
        """分岐比を返す"""
        return self.alpha / self.beta if self.beta > 0 else np.inf

    def get_name(self) -> str:
        return "指数減衰"

    def get_params_dict(self) -> dict:
        return {"α": self.alpha, "β": self.beta}


class PowerLawKernel(Kernel):
    """
    べき乗則カーネル (Omori型): α / (t + c)^(β+1)

    長時間記憶を持つ。地震の余震モデル（Omori's law）で使用。
    """

    def __init__(self, alpha: float, beta: float, c: float = 0.1):
        """
        Args:
            alpha: 励起の強さ
            beta: 減衰指数（大きいほど速く減衰）
            c: 特異点回避のためのシフトパラメータ
        """
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t)
        result = np.where(
            t >= 0,
            self.alpha / np.power(t + self.c, self.beta + 1),
            0.0
        )
        return float(result) if result.ndim == 0 else result

    def get_upper_bound(self) -> float:
        # t=0での値が最大
        return self.alpha / np.power(self.c, self.beta + 1)

    def check_stability(self) -> bool:
        # β > 0 で積分が収束
        # さらに、実用上は積分値 < 1 が望ましい
        integral = self.get_integral()
        return self.beta > 0 and integral < 1.0

    def get_integral(self, T: float = np.inf) -> float:
        if self.beta == 0:
            if T == np.inf:
                return np.inf
            return self.alpha * np.log((T + self.c) / self.c)

        if T == np.inf:
            # β > 0 のとき収束
            return (self.alpha / self.beta) * np.power(self.c, -self.beta)

        return (self.alpha / self.beta) * (
            np.power(self.c, -self.beta) - np.power(T + self.c, -self.beta)
        )

    def get_name(self) -> str:
        return "べき乗則 (Omori)"

    def get_params_dict(self) -> dict:
        return {"α": self.alpha, "β": self.beta, "c": self.c}


class SumOfExponentialsKernel(Kernel):
    """
    複数指数の和: Σ αᵢ * exp(-βᵢ * t)

    複数の時間スケールを表現。短期と長期の影響を同時にモデル化。
    """

    def __init__(self, alphas: list[float], betas: list[float]):
        """
        Args:
            alphas: 各成分の励起の強さのリスト
            betas: 各成分の減衰率のリスト
        """
        assert len(alphas) == len(betas), "alphas and betas must have same length"
        self.alphas = np.array(alphas)
        self.betas = np.array(betas)
        self.n_components = len(alphas)

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t)
        result = np.zeros_like(t, dtype=float)

        for alpha, beta in zip(self.alphas, self.betas):
            result = result + np.where(
                t >= 0,
                alpha * np.exp(-beta * t),
                0.0
            )

        return float(result) if result.ndim == 0 else result

    def get_upper_bound(self) -> float:
        return np.sum(self.alphas)

    def check_stability(self) -> bool:
        # 各成分の分岐比の和 < 1
        branching_ratios = self.alphas / self.betas
        return np.sum(branching_ratios) < 1.0

    def get_integral(self, T: float = np.inf) -> float:
        if T == np.inf:
            return np.sum(self.alphas / self.betas)

        integral = 0.0
        for alpha, beta in zip(self.alphas, self.betas):
            integral += (alpha / beta) * (1 - np.exp(-beta * T))
        return integral

    def get_branching_ratio(self) -> float:
        """総分岐比を返す"""
        return np.sum(self.alphas / self.betas)

    def get_name(self) -> str:
        return f"複数指数 ({self.n_components}成分)"

    def get_params_dict(self) -> dict:
        params = {}
        for i, (alpha, beta) in enumerate(zip(self.alphas, self.betas), 1):
            params[f"α{i}"] = alpha
            params[f"β{i}"] = beta
        return params


class GaussianKernel(Kernel):
    """
    ガウスカーネル: α * exp(-(t-μ)² / (2σ²))

    滑らかな山型の影響。特定の遅延時間にピーク。
    """

    def __init__(self, alpha: float, mu: float = 1.0, sigma: float = 0.5):
        """
        Args:
            alpha: 励起の強さ
            mu: ピークの位置（遅延時間）
            sigma: 広がり（標準偏差）
        """
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t)
        result = np.where(
            t >= 0,
            self.alpha * np.exp(-np.power(t - self.mu, 2) / (2 * self.sigma**2)),
            0.0
        )
        return float(result) if result.ndim == 0 else result

    def get_upper_bound(self) -> float:
        # μでの値が最大（μ >= 0と仮定）
        if self.mu >= 0:
            return self.alpha
        # μ < 0 の場合、t=0での値
        return self.alpha * np.exp(-self.mu**2 / (2 * self.sigma**2))

    def check_stability(self) -> bool:
        # 積分値 < 1 を条件とする
        integral = self.get_integral()
        return integral < 1.0

    def get_integral(self, T: float = np.inf) -> float:
        # ガウス分布のCDFを使用
        sqrt2 = np.sqrt(2)

        if T == np.inf:
            # 全体の積分: α * σ * sqrt(2π) * [1 - Φ(-μ/σ)]
            # Φ(x) = (1 + erf(x/sqrt(2))) / 2
            cdf_at_neg_mu = 0.5 * (1 + erf(-self.mu / (self.sigma * sqrt2)))
            return self.alpha * self.sigma * np.sqrt(2 * np.pi) * (1 - cdf_at_neg_mu)

        # 0からTまでの積分
        cdf_at_T = 0.5 * (1 + erf((T - self.mu) / (self.sigma * sqrt2)))
        cdf_at_0 = 0.5 * (1 + erf(-self.mu / (self.sigma * sqrt2)))
        return self.alpha * self.sigma * np.sqrt(2 * np.pi) * (cdf_at_T - cdf_at_0)

    def get_name(self) -> str:
        return "ガウス"

    def get_params_dict(self) -> dict:
        return {"α": self.alpha, "μ": self.mu, "σ": self.sigma}


class RectangularKernel(Kernel):
    """
    矩形カーネル: α (0 ≤ t ≤ T)

    一定期間だけ一定の励起。シンプルで解釈しやすい。
    """

    def __init__(self, alpha: float, duration: float):
        """
        Args:
            alpha: 励起の強さ
            duration: 励起が続く期間
        """
        self.alpha = alpha
        self.duration = duration

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t)
        result = np.where(
            (t >= 0) & (t <= self.duration),
            self.alpha,
            0.0
        )
        return float(result) if result.ndim == 0 else result

    def get_upper_bound(self) -> float:
        return self.alpha

    def check_stability(self) -> bool:
        # 積分値 = α * duration < 1
        return self.alpha * self.duration < 1.0

    def get_integral(self, T: float = np.inf) -> float:
        if T >= self.duration:
            return self.alpha * self.duration
        return self.alpha * T

    def get_name(self) -> str:
        return "矩形"

    def get_params_dict(self) -> dict:
        return {"α": self.alpha, "T": self.duration}


# カーネルファクトリー関数
def create_kernel(kernel_type: str, **params) -> Kernel:
    """
    カーネルを生成するファクトリー関数

    Args:
        kernel_type: カーネルの種類
        **params: カーネルのパラメータ

    Returns:
        Kernelインスタンス
    """
    kernel_map = {
        "exponential": ExponentialKernel,
        "power_law": PowerLawKernel,
        "sum_exp": SumOfExponentialsKernel,
        "gaussian": GaussianKernel,
        "rectangular": RectangularKernel,
    }

    if kernel_type not in kernel_map:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel_map[kernel_type](**params)
