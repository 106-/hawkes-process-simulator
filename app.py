"""
Hawkes Process Interactive Visualization App

A Streamlit app for exploring univariate Hawkes processes with
various kernel functions through interactive simulation and visualization.
"""

import streamlit as st
import numpy as np
from src.hawkes_process import HawkesProcess
from src.kernels import (
    ExponentialKernel, PowerLawKernel, SumOfExponentialsKernel,
    GaussianKernel, RectangularKernel
)
from src.visualization import (
    plot_kernels_comparison,
    plot_kernel_properties
)


# Page configuration
st.set_page_config(
    page_title="Hawkes Process Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# キャッシュを削除 - カーネルオブジェクトのハッシュ問題を回避
# シミュレーションは高速なのでキャッシュ不要


def create_kernel_from_ui():
    """UIの設定からカーネルオブジェクトを作成"""

    kernel_type = st.sidebar.selectbox(
        "カーネルの種類",
        ["指数減衰", "べき乗則 (Omori)", "複数指数", "ガウス", "矩形"],
        help="励起関数の種類を選択"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {kernel_type}パラメータ")

    if kernel_type == "指数減衰":
        alpha = st.sidebar.slider(
            "α (励起強度)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.05,
            help="イベント発生時の強度ジャンプ"
        )
        beta = st.sidebar.slider(
            "β (減衰率)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="励起が消える速度"
        )
        kernel = ExponentialKernel(alpha, beta)

    elif kernel_type == "べき乗則 (Omori)":
        alpha = st.sidebar.slider(
            "α (励起強度)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.05
        )
        beta = st.sidebar.slider(
            "β (減衰指数)",
            min_value=0.1, max_value=3.0, value=1.0, step=0.1,
            help="大きいほど速く減衰"
        )
        c = st.sidebar.slider(
            "c (シフト)",
            min_value=0.01, max_value=1.0, value=0.1, step=0.01,
            help="特異点回避パラメータ"
        )
        kernel = PowerLawKernel(alpha, beta, c)

    elif kernel_type == "複数指数":
        st.sidebar.markdown("**短期成分**")
        alpha1 = st.sidebar.slider(
            "α₁", min_value=0.1, max_value=1.0, value=0.3, step=0.05
        )
        beta1 = st.sidebar.slider(
            "β₁", min_value=0.5, max_value=5.0, value=2.0, step=0.1
        )

        st.sidebar.markdown("**長期成分**")
        alpha2 = st.sidebar.slider(
            "α₂", min_value=0.1, max_value=1.0, value=0.2, step=0.05
        )
        beta2 = st.sidebar.slider(
            "β₂", min_value=0.1, max_value=2.0, value=0.5, step=0.1
        )
        kernel = SumOfExponentialsKernel([alpha1, alpha2], [beta1, beta2])

    elif kernel_type == "ガウス":
        alpha = st.sidebar.slider(
            "α (励起強度)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.05
        )
        mu_param = st.sidebar.slider(
            "μ (ピーク位置)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="励起のピークまでの遅延時間"
        )
        sigma = st.sidebar.slider(
            "σ (広がり)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            help="励起の時間的な広がり"
        )
        kernel = GaussianKernel(alpha, mu_param, sigma)

    else:  # 矩形
        alpha = st.sidebar.slider(
            "α (励起強度)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.05
        )
        duration = st.sidebar.slider(
            "T (持続時間)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help="励起が持続する期間"
        )
        kernel = RectangularKernel(alpha, duration)

    return kernel


def main():
    """Main application."""

    # Title
    st.title("📈 Hawkes Process Simulator")
    st.markdown("""
    単変量Hawkesプロセスのシミュレーションと可視化ツール。
    様々なカーネル関数を選択して、自己励起過程の挙動を体験できます。
    """)

    # Create placeholders for button and stability check at the top
    button_placeholder = st.sidebar.empty()
    stability_placeholder = st.sidebar.empty()

    # Sidebar - Parameters
    st.sidebar.header("⚙️ パラメータ設定")

    # Baseline intensity
    mu = st.sidebar.slider(
        "μ (ベースライン強度)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="自発的に発生するイベントの率"
    )

    # Kernel selection
    kernel = create_kernel_from_ui()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### シミュレーション設定")

    T_max = st.sidebar.slider(
        "シミュレーション時間",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="シミュレーションの終了時刻"
    )

    seed = st.sidebar.number_input(
        "ランダムシード",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="再現性のためのシード値"
    )

    st.sidebar.markdown("---")

    # Display current parameters
    st.sidebar.markdown("### 現在の設定")
    st.sidebar.markdown(f"**カーネル:** {kernel.get_name()}")

    params = kernel.get_params_dict()
    for key, value in params.items():
        st.sidebar.markdown(f"- **{key}** = {value:.2f}")

    branching_ratio = kernel.get_integral()
    st.sidebar.markdown(f"- **分岐比** = {branching_ratio:.3f}")

    # Now populate the placeholders at the top with button and stability check
    with button_placeholder:
        simulate_button = st.button("🎲 シミュレーション実行", type="primary")

    with stability_placeholder:
        hp = HawkesProcess(mu, kernel=kernel)
        if hp.check_stability():
            st.success(f"✓ 安定なプロセス (n = {branching_ratio:.3f} < 1)")
        else:
            st.error("⚠️ 安定条件違反！")
            st.warning("プロセスが発散する可能性があります")

    # Mathematical formula
    st.sidebar.markdown("---")
    with st.sidebar.expander("📐 強度関数の式"):
        st.latex(r"\lambda(t) = \mu + \sum_{t_i < t} \varphi(t - t_i)")
        st.markdown("""
        - **μ**: ベースライン強度
        - **φ(t)**: カーネル関数（励起関数）
        - **t_i**: 過去のイベント時刻
        """)

    # Main content
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 シミュレーション", "🔬 カーネル詳細", "📈 カーネル比較"])

    with tab1:
        # ボタンが押されたときか初回のみシミュレーション実行
        if 'last_params' not in st.session_state or simulate_button:
            with st.spinner('シミュレーション実行中...'):
                hp = HawkesProcess(mu, kernel=kernel)
                max_events = 10000  # イベント数上限
                max_iterations = 100000  # ループ反復回数上限（無限ループ防止）
                event_times = hp.simulate(T_max, seed,
                                         max_events=max_events,
                                         max_iterations=max_iterations)
                time_grid, intensity = hp.get_intensity_trace(event_times, T_max, n_points=1000)

                # 結果をsession_stateに保存
                st.session_state.last_params = (kernel, mu, T_max, seed)
                st.session_state.event_times = event_times
                st.session_state.time_grid = time_grid
                st.session_state.intensity = intensity
                st.session_state.max_events = max_events

        # session_stateから結果を取得
        event_times = st.session_state.event_times
        time_grid = st.session_state.time_grid
        intensity = st.session_state.intensity
        kernel_s, mu_s, T_max_s, seed_s = st.session_state.last_params

        # パラメータが変更されているかチェック
        kernel_params_changed = (kernel.get_params_dict() != kernel_s.get_params_dict() or
                                kernel.get_name() != kernel_s.get_name())
        params_changed = (kernel_params_changed or mu != mu_s or T_max != T_max_s or seed != seed_s)

        if params_changed:
            st.warning("⚠️ パラメータが変更されています。「🎲 シミュレーション実行」ボタンをクリックして更新してください。")

        # 発散警告
        if len(event_times) >= st.session_state.max_events:
            st.error(f"⚠️ イベント数が上限（{st.session_state.max_events}）に達しました！")
            st.warning("プロセスが発散しています。パラメータを調整してください（αを小さく、またはβを大きく）")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("イベント数", len(event_times))

        with col2:
            avg_rate = len(event_times) / T_max_s if T_max_s > 0 else 0
            st.metric("平均レート", f"{avg_rate:.2f}")

        with col3:
            max_intensity = np.max(intensity) if len(intensity) > 0 else 0
            st.metric("最大強度", f"{max_intensity:.2f}")

        with col4:
            branching = kernel_s.get_integral()
            st.metric("分岐比", f"{branching:.3f}")

        # Main plot - Need to adapt for different kernels
        st.markdown("### 強度関数とイベント発生")

        # Create custom plot since we now have different kernels
        import plotly.graph_objects as go

        fig = go.Figure()

        # Intensity curve
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=intensity,
            mode='lines',
            name='λ(t) 強度関数',
            line=dict(color='#2E86DE', width=2.5),
            hovertemplate='時刻: %{x:.2f}<br>強度: %{y:.3f}<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(46, 134, 222, 0.1)'
        ))

        # Baseline
        fig.add_hline(
            y=mu_s,
            line_dash="dash",
            line_color="#10AC84",
            line_width=2,
            annotation_text=f"ベースライン μ = {mu_s:.2f}",
            annotation_position="right"
        )

        # Event markers
        if len(event_times) > 0:
            intensity_at_events = []
            for t in event_times:
                idx = np.argmin(np.abs(time_grid - t))
                intensity_at_events.append(intensity[idx])

            fig.add_trace(go.Scatter(
                x=event_times,
                y=intensity_at_events,
                mode='markers',
                name='イベント発生',
                marker=dict(size=10, color='#EE5A6F', symbol='x', line=dict(width=2)),
                hovertemplate='イベント時刻: %{x:.2f}<br>強度: %{y:.3f}<extra></extra>'
            ))

            for t in event_times:
                fig.add_vline(x=t, line_dash="dot", line_color="#EE5A6F", line_width=1, opacity=0.4)

        fig.update_layout(
            title=f'Hawkes Process ({kernel_s.get_name()}カーネル)',
            xaxis_title="時刻 (t)",
            yaxis_title="強度 λ(t)",
            hovermode='x unified',
            showlegend=True,
            height=500,
            plot_bgcolor='white',
            xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
            yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)')
        )

        st.plotly_chart(fig, width='stretch')

        # Event times table
        if len(event_times) > 0:
            with st.expander("📋 イベント発生時刻の詳細"):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**最初の10イベント:**")
                    for i, t in enumerate(event_times[:10]):
                        st.text(f"{i+1}. t = {t:.3f}")

                with col_b:
                    if len(event_times) > 1:
                        inter_event_times = np.diff(event_times)
                        st.markdown("**イベント間隔の統計:**")
                        st.text(f"平均: {np.mean(inter_event_times):.3f}")
                        st.text(f"最小: {np.min(inter_event_times):.3f}")
                        st.text(f"最大: {np.max(inter_event_times):.3f}")
        else:
            st.info("イベントが発生しませんでした。μを大きくするか、T_maxを増やしてください。")

    with tab2:
        st.markdown("### カーネル関数の詳細分析")
        st.markdown(f"**使用中のカーネル:** {kernel.get_name()}")

        # Kernel properties visualization
        fig_props = plot_kernel_properties(kernel, t_max=20.0)
        st.plotly_chart(fig_props, width='stretch')

        # Kernel statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            initial_jump = kernel.get_upper_bound()
            st.metric("初期ジャンプ", f"{initial_jump:.3f}")

        with col2:
            integral_val = kernel.get_integral()
            st.metric("総積分値 (分岐比)", f"{integral_val:.3f}")

        with col3:
            stability = "安定" if kernel.check_stability() else "不安定"
            st.metric("安定性", stability)

    with tab3:
        st.markdown("### カーネル関数の比較")

        st.markdown("""
        異なるカーネル関数の形状と性質を比較します。
        分岐比（n = カーネルの積分値）が表示されます。
        """)

        # Create comparison kernels
        comparison_kernels = [
            ExponentialKernel(alpha=0.5, beta=1.0),
            PowerLawKernel(alpha=0.5, beta=1.0, c=0.1),
            GaussianKernel(alpha=0.5, mu=1.0, sigma=0.5),
            RectangularKernel(alpha=0.25, duration=2.0),
        ]

        labels = [
            "指数減衰 (α=0.5, β=1.0)",
            "べき乗則 (α=0.5, β=1.0, c=0.1)",
            "ガウス (α=0.5, μ=1.0, σ=0.5)",
            "矩形 (α=0.25, T=2.0)"
        ]

        fig_comparison = plot_kernels_comparison(comparison_kernels, t_max=10.0, labels=labels)
        st.plotly_chart(fig_comparison, width='stretch')

        # Comparison table
        st.markdown("#### カーネル特性の比較表")

        comparison_data = []
        for kernel_cmp in comparison_kernels:
            comparison_data.append({
                "カーネル": kernel_cmp.get_name(),
                "分岐比": f"{kernel_cmp.get_integral():.3f}",
                "安定性": "✓" if kernel_cmp.check_stability() else "✗",
                "特徴": get_kernel_description(kernel_cmp)
            })

        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.table(df)


def get_kernel_description(kernel) -> str:
    """カーネルの特徴を説明"""
    name = kernel.get_name()

    descriptions = {
        "指数減衰": "短時間記憶、最も一般的",
        "べき乗則 (Omori)": "長時間記憶、地震余震モデル",
        "複数指数 (2成分)": "複数時間スケール",
        "ガウス": "遅延ピーク、滑らか",
        "矩形": "一定期間の一定励起"
    }

    return descriptions.get(name, "")


if __name__ == "__main__":
    main()
