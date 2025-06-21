import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg


class ARModelEstimator:
    """AR模型参数估计器"""

    def __init__(self):
        self.coefficients = None
        self.noise_variance = None
        self.fitted_values = None
        self.original_data = None
        self.order = None

    def estimate_parameters(self, data, order):
        """
        使用Yule-Walker方程估计AR模型参数

        参数:
        data: 时间序列数据 (numpy array)
        order: AR模型的阶数 (int)

        返回:
        coefficients: AR系数 (numpy array)
        noise_variance: 白噪声方差 (float)
        """
        self.original_data = data
        self.order = order

        # 计算自相关函数
        n = len(data)
        mean = np.mean(data)
        centered_data = data - mean

        # 计算自协方差函数
        autocov = np.zeros(order + 1)
        for k in range(order + 1):
            if k == 0:
                autocov[k] = np.var(centered_data, ddof=0)
            else:
                autocov[k] = np.mean(centered_data[:-k] * centered_data[k:])

        # 构建Yule-Walker方程的系数矩阵
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = autocov[abs(i - j)]

        # 构建右侧向量
        r = autocov[1:order + 1]

        # 求解Yule-Walker方程
        try:
            self.coefficients = linalg.solve(R, r)
        except linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘解
            self.coefficients = linalg.lstsq(R, r)[0]

        # 计算白噪声方差
        self.noise_variance = autocov[0] - np.dot(self.coefficients, r)

        return self.coefficients, self.noise_variance

    def fit_model(self):
        """
        使用估计的参数拟合AR模型

        返回:
        fitted_values: 拟合值 (numpy array)
        """
        if self.coefficients is None or self.original_data is None:
            raise ValueError("请先估计模型参数")

        n = len(self.original_data)
        self.fitted_values = np.zeros(n)

        # 前p个值使用原始数据
        self.fitted_values[:self.order] = self.original_data[:self.order]

        # 从第p+1个值开始使用AR模型预测
        for t in range(self.order, n):
            prediction = 0
            for i in range(self.order):
                prediction += self.coefficients[i] * self.original_data[t - 1 - i]
            self.fitted_values[t] = prediction

        return self.fitted_values

    def get_model_summary(self):
        """
        获取模型摘要信息

        返回:
        summary: 包含模型参数的字典
        """
        if self.coefficients is None:
            return None

        summary = {
            'order': self.order,
            'coefficients': self.coefficients,
            'noise_variance': self.noise_variance,
            'coefficient_names': [f'φ{i+1}' for i in range(self.order)]
        }

        return summary


def load_data_from_file(uploaded_file):
    """
    从上传的文件中加载时间序列数据

    参数:
    uploaded_file: Streamlit上传的文件对象

    返回:
    data: 时间序列数据 (numpy array)
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # 假设第一列是时间序列数据
            data = df.iloc[:, 0].values
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            data = np.array([float(x.strip()) for x in content.split() if x.strip()])
        else:
            st.error("不支持的文件格式。请上传CSV或TXT文件。")
            return None

        return data
    except Exception as e:
        st.error(f"文件读取错误: {str(e)}")
        return None


def parse_manual_input(input_text):
    """
    解析手动输入的时间序列数据

    参数:
    input_text: 用户输入的文本

    返回:
    data: 时间序列数据 (numpy array)
    """
    try:
        # 支持逗号、空格、换行分隔
        input_text = input_text.replace(',', ' ').replace('\n', ' ')
        data = np.array([float(x.strip()) for x in input_text.split() if x.strip()])
        return data
    except Exception as e:
        st.error(f"数据解析错误: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="时序模型参数估计器",
        page_icon="📈",
        layout="wide"
    )
    
    # 设置Matplotlib中文字体支持
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    st.title("📈 时序模型参数估计器")
    st.markdown("---")

    # 侧边栏配置
    st.sidebar.header("配置选项")

    # 数据输入方式选择
    input_method = st.sidebar.radio(
        "选择数据输入方式:",
        ["文件上传", "手动输入", "示例数据"]
    )

    data = None

    if input_method == "文件上传":
        uploaded_file = st.sidebar.file_uploader(
            "上传时间序列数据文件",
            type=['csv', 'txt'],
            help="支持CSV和TXT格式文件"
        )
        if uploaded_file is not None:
            data = load_data_from_file(uploaded_file)

    elif input_method == "手动输入":
        input_text = st.sidebar.text_area(
            "输入时间序列数据:",
            placeholder="请输入数值，用逗号、空格或换行分隔\n例如: 1.2, 2.3, 3.4, 4.5",
            height=150
        )
        if input_text.strip():
            data = parse_manual_input(input_text)

    else:  # 示例数据
        st.sidebar.info("使用AR(2)模型生成的示例数据")
        np.random.seed(42)
        n = 100
        noise = np.random.normal(0, 1, n)
        data = np.zeros(n)
        data[0] = noise[0]
        data[1] = noise[1]
        for t in range(2, n):
            data[t] = 0.6 * data[t-1] - 0.3 * data[t-2] + noise[t]

    if data is not None and len(data) > 0:
        # 数据预览
        st.subheader("📊 数据预览")
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data, 'b-', linewidth=1.5, label='原始数据')
            ax.set_xlabel('时间')
            ax.set_ylabel('数值')
            ax.set_title('时间序列数据')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.metric("数据点数量", len(data))
            st.metric("均值", f"{np.mean(data):.4f}")
            st.metric("标准差", f"{np.std(data):.4f}")
            st.metric("最小值", f"{np.min(data):.4f}")
            st.metric("最大值", f"{np.max(data):.4f}")

        # AR模型参数设置
        st.subheader("⚙️ 模型参数设置")
        max_order = min(20, len(data) // 4)  # 限制最大阶数
        order = st.slider(
            "选择AR模型阶数:",
            min_value=1,
            max_value=max_order,
            value=min(2, max_order),
            help=f"建议阶数不超过数据长度的1/4 (当前最大: {max_order})"
        )

        if st.button("🔍 估计模型参数", type="primary"):
            with st.spinner("正在估计模型参数..."):
                # 创建AR模型估计器
                estimator = ARModelEstimator()

                try:
                    # 估计参数
                    coefficients, noise_variance = estimator.estimate_parameters(data, order)

                    # 拟合模型
                    fitted_values = estimator.fit_model()

                    # 获取模型摘要
                    summary = estimator.get_model_summary()

                    # 显示结果
                    st.success("模型参数估计完成！")

                    # 参数展示
                    st.subheader("📋 模型参数")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**AR系数:**")
                        coef_df = pd.DataFrame({
                            '参数': summary['coefficient_names'],
                            '系数值': summary['coefficients']
                        })
                        st.dataframe(coef_df, use_container_width=True)

                    with col2:
                        st.metric("白噪声方差 (σ²)", f"{summary['noise_variance']:.6f}")
                        st.metric("模型阶数", summary['order'])

                        # 计算拟合优度指标
                        mse = np.mean((data[order:] - fitted_values[order:]) ** 2)
                        st.metric("均方误差 (MSE)", f"{mse:.6f}")

                    # 可视化对比
                    st.subheader("📈 模型拟合效果对比")

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                    # 上图：完整对比
                    ax1.plot(data, 'b-', linewidth=2, label='原始数据', alpha=0.8)
                    ax1.plot(fitted_values, 'r--', linewidth=2, label='拟合数据', alpha=0.8)
                    ax1.set_xlabel('时间')
                    ax1.set_ylabel('数值')
                    ax1.set_title('原始数据 vs 拟合数据 (完整视图)')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                    # 下图：残差
                    residuals = data - fitted_values
                    ax2.plot(residuals, 'g-', linewidth=1.5, alpha=0.7)
                    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
                    ax2.set_xlabel('时间')
                    ax2.set_ylabel('残差')
                    ax2.set_title('拟合残差')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # 局部放大视图
                    if len(data) > 50:
                        st.subheader("🔍 局部拟合效果 (前50个数据点)")
                        fig, ax = plt.subplots(figsize=(12, 5))

                        subset_range = slice(0, 50)
                        ax.plot(data[subset_range], 'bo-', linewidth=2, markersize=4,
                               label='原始数据', alpha=0.8)
                        ax.plot(fitted_values[subset_range], 'rs--', linewidth=2, markersize=4,
                               label='拟合数据', alpha=0.8)
                        ax.set_xlabel('时间')
                        ax.set_ylabel('数值')
                        ax.set_title('局部拟合效果对比')
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                        st.pyplot(fig)
                        plt.close()

                    # 模型诊断
                    st.subheader("🔬 模型诊断")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # 残差统计
                        residual_mean = np.mean(residuals)
                        residual_std = np.std(residuals)
                        st.metric("残差均值", f"{residual_mean:.6f}")
                        st.metric("残差标准差", f"{residual_std:.6f}")

                    with col2:
                        # 拟合优度
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((data - np.mean(data)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        st.metric("R² 决定系数", f"{r_squared:.4f}")

                        # AIC信息准则 (简化版)
                        n = len(data)
                        aic = n * np.log(ss_res / n) + 2 * order
                        st.metric("AIC", f"{aic:.2f}")

                    with col3:
                        # 预测区间
                        prediction_std = np.sqrt(noise_variance)
                        st.metric("预测标准误", f"{prediction_std:.4f}")
                        st.metric("95%置信区间", f"±{1.96 * prediction_std:.4f}")

                except Exception as e:
                    st.error(f"模型估计过程中出现错误: {str(e)}")
                    st.info("请检查数据质量或尝试调整模型阶数。")

    else:
        st.info("👆 请在左侧选择数据输入方式并提供时间序列数据")

        # 显示使用说明
        st.subheader("📖 使用说明")
        st.markdown("""
        ### 功能特点
        - **AR模型参数估计**: 使用Yule-Walker方程计算AR模型系数和白噪声方差
        - **模型拟合与可视化**: 生成拟合值并与原始数据进行对比展示
        - **多种数据输入方式**: 支持文件上传、手动输入和示例数据
        - **模型诊断**: 提供残差分析、拟合优度等诊断指标

        ### 数据格式要求
        - **CSV文件**: 第一列为时间序列数据
        - **TXT文件**: 数值用空格或换行分隔
        - **手动输入**: 数值用逗号、空格或换行分隔

        ### 模型说明
        AR(p)模型形式: X(t) = φ₁X(t-1) + φ₂X(t-2) + ... + φₚX(t-p) + ε(t)

        其中:
        - φᵢ 为AR系数
        - ε(t) 为白噪声项，方差为σ²
        - p 为模型阶数
        """)


if __name__ == "__main__":
    main()
