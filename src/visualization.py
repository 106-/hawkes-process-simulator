"""
Visualization Module for Hawkes Process

Creates interactive Plotly visualizations for Hawkes process simulations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def plot_intensity_timeline(
    event_times: np.ndarray,
    time_grid: np.ndarray,
    intensity_values: np.ndarray,
    mu: float,
    alpha: float,
    beta: float
) -> go.Figure:
    """
    Create main visualization showing events and intensity function.

    Args:
        event_times: Array of event occurrence times
        time_grid: Time points for intensity plot
        intensity_values: Intensity values at each time point
        mu: Baseline intensity parameter
        alpha: Excitation parameter
        beta: Decay parameter

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Plot intensity curve
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=intensity_values,
        mode='lines',
        name='λ(t) 強度関数',
        line=dict(color='#2E86DE', width=2.5),
        hovertemplate='時刻: %{x:.2f}<br>強度: %{y:.3f}<extra></extra>',
        fill='tozeroy',
        fillcolor='rgba(46, 134, 222, 0.1)'
    ))

    # Plot baseline as horizontal line
    fig.add_hline(
        y=mu,
        line_dash="dash",
        line_color="#10AC84",
        line_width=2,
        annotation_text=f"ベースライン μ = {mu:.2f}",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="#10AC84"
    )

    # Plot event markers
    if len(event_times) > 0:
        # Compute intensity at event times for y-coordinates
        intensity_at_events = []
        for t in event_times:
            idx = np.argmin(np.abs(time_grid - t))
            intensity_at_events.append(intensity_values[idx])

        fig.add_trace(go.Scatter(
            x=event_times,
            y=intensity_at_events,
            mode='markers',
            name='イベント発生',
            marker=dict(
                size=10,
                color='#EE5A6F',
                symbol='x',
                line=dict(width=2)
            ),
            hovertemplate='イベント時刻: %{x:.2f}<br>強度: %{y:.3f}<extra></extra>'
        ))

        # Add vertical lines at events
        for t in event_times:
            fig.add_vline(
                x=t,
                line_dash="dot",
                line_color="#EE5A6F",
                line_width=1,
                opacity=0.4
            )

    # Layout configuration
    fig.update_layout(
        title={
            'text': 'Hawkes Process: 強度関数とイベント',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'sans-serif'}
        },
        xaxis_title="時刻 (t)",
        yaxis_title="強度 λ(t)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        height=500,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False
        )
    )

    return fig


def plot_kernel_decay(alpha: float, beta: float, t_max: float = 10.0) -> go.Figure:
    """
    Visualize the exponential decay kernel shape.

    Shows how a single event's contribution decays over time:
    kernel(t) = α * exp(-β * t)

    Args:
        alpha: Excitation parameter
        beta: Decay rate
        t_max: Maximum time to show

    Returns:
        Plotly Figure object
    """
    t = np.linspace(0, t_max, 500)
    kernel = alpha * np.exp(-beta * t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=kernel,
        mode='lines',
        name='減衰カーネル',
        line=dict(color='#8854D0', width=3),
        fill='tozeroy',
        fillcolor='rgba(136, 84, 208, 0.2)',
        hovertemplate='時間差: %{x:.2f}<br>寄与: %{y:.3f}<extra></extra>'
    ))

    # Mark half-life point
    half_life = np.log(2) / beta
    if half_life < t_max:
        fig.add_vline(
            x=half_life,
            line_dash="dash",
            line_color="#F79F1F",
            annotation_text=f"半減期: {half_life:.2f}",
            annotation_position="top"
        )

    fig.update_layout(
        title={
            'text': f'減衰カーネル: α={alpha:.2f}, β={beta:.2f}',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="イベント後の経過時間",
        yaxis_title="強度への寄与",
        height=400,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)')
    )

    return fig


def plot_parameter_comparison(
    simulations: list[dict],
    T_max: float
) -> go.Figure:
    """
    Compare multiple simulations with different parameters.

    Args:
        simulations: List of dicts containing simulation results
        T_max: Maximum simulation time

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=len(simulations),
        cols=1,
        subplot_titles=[f"μ={s['mu']}, α={s['alpha']}, β={s['beta']}"
                       for s in simulations],
        vertical_spacing=0.1
    )

    colors = ['#2E86DE', '#EE5A6F', '#10AC84', '#F79F1F', '#8854D0']

    for i, sim in enumerate(simulations):
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=sim['time_grid'],
                y=sim['intensity'],
                mode='lines',
                name=f"Sim {i+1}",
                line=dict(color=color, width=2)
            ),
            row=i+1, col=1
        )

    fig.update_layout(
        height=300 * len(simulations),
        showlegend=False,
        title_text="パラメータ比較"
    )

    return fig


def plot_kernels_comparison(kernels: list, t_max: float = 10.0, labels: Optional[list[str]] = None) -> go.Figure:
    """
    複数のカーネル関数を比較する可視化
    
    Args:
        kernels: Kernelオブジェクトのリスト
        t_max: 表示する最大時間
        labels: 各カーネルのラベル（Noneの場合はカーネル名を使用）
    
    Returns:
        Plotly Figure object
    """
    t = np.linspace(0, t_max, 500)
    
    colors = ['#2E86DE', '#EE5A6F', '#10AC84', '#F79F1F', '#8854D0', '#0ABDE3']
    
    fig = go.Figure()
    
    for i, kernel in enumerate(kernels):
        kernel_values = kernel.evaluate(t)
        label = labels[i] if labels and i < len(labels) else kernel.get_name()
        color = colors[i % len(colors)]
        
        # カーネル曲線
        fig.add_trace(go.Scatter(
            x=t,
            y=kernel_values,
            mode='lines',
            name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f'{label}<br>時間: %{{x:.2f}}<br>値: %{{y:.3f}}<extra></extra>'
        ))
        
        # 積分値（分岐比）をアノテーションで表示
        integral = kernel.get_integral()
        if np.isfinite(integral):
            fig.add_annotation(
                x=t_max * 0.95,
                y=kernel_values[-1] if len(kernel_values) > 0 else 0,
                text=f"n={integral:.3f}",
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor='right'
            )
    
    fig.update_layout(
        title="カーネル関数の比較",
        xaxis_title="経過時間",
        yaxis_title="カーネル値 φ(t)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        height=500,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)')
    )
    
    return fig


def plot_kernel_properties(kernel, t_max: float = 10.0) -> go.Figure:
    """
    単一カーネルの詳細なプロパティを可視化
    
    Args:
        kernel: Kernelオブジェクト
        t_max: 表示する最大時間
    
    Returns:
        Plotly Figure object
    """
    t = np.linspace(0, t_max, 500)
    kernel_values = kernel.evaluate(t)
    
    # サブプロット作成
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{kernel.get_name()}カーネル',
            '累積積分（分岐比への寄与）'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # カーネル関数
    fig.add_trace(
        go.Scatter(
            x=t,
            y=kernel_values,
            mode='lines',
            name='φ(t)',
            line=dict(color='#2E86DE', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 222, 0.2)',
            hovertemplate='時間: %{x:.2f}<br>値: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 累積積分
    cumulative = np.array([kernel.get_integral(ti) for ti in t])
    total_integral = kernel.get_integral()
    
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cumulative,
            mode='lines',
            name='∫₀ᵗ φ(s)ds',
            line=dict(color='#10AC84', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 172, 132, 0.2)',
            hovertemplate='時間: %{x:.2f}<br>累積: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 総積分値（分岐比）を水平線で表示
    if np.isfinite(total_integral):
        fig.add_hline(
            y=total_integral,
            line_dash="dash",
            line_color="#EE5A6F",
            annotation_text=f"総積分値 n = {total_integral:.3f}",
            annotation_position="right",
            row=2, col=1
        )
    
    # レイアウト更新
    fig.update_xaxes(title_text="経過時間", row=1, col=1)
    fig.update_xaxes(title_text="経過時間", row=2, col=1)
    fig.update_yaxes(title_text="φ(t)", row=1, col=1)
    fig.update_yaxes(title_text="累積積分", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # パラメータ情報をアノテーションで追加
    params = kernel.get_params_dict()
    params_text = ", ".join([f"{k}={v:.2f}" for k, v in params.items()])
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=f"パラメータ: {params_text}",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        xanchor='left',
        yanchor='top'
    )
    
    return fig
