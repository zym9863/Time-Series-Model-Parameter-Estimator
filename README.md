[English Version (README_EN.md)](README_EN.md)

# 时序模型参数估计器

一个基于Python和Streamlit的时间序列AR模型参数估计工具，提供直观的Web界面和强大的分析功能。

## 🚀 功能特点

### 核心功能
- **AR模型参数估计**: 使用Yule-Walker方程精确计算AR模型系数和白噪声方差
- **模型拟合与可视化**: 生成拟合值并与原始数据进行直观对比
- **多种数据输入方式**: 支持文件上传、手动输入和示例数据
- **模型诊断**: 提供残差分析、拟合优度等专业诊断指标

### 界面特色
- 🎨 现代化的Web界面设计
- 📊 交互式图表展示
- 📈 实时参数计算
- 🔍 局部拟合效果放大查看
- 📋 详细的模型摘要报告

## 🛠️ 技术栈

- **Python 3.12+**: 核心编程语言
- **Streamlit**: Web界面框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib**: 数据可视化
- **SciPy**: 科学计算
- **uv**: 现代Python包管理器

## 📦 安装与运行

### 环境要求
- Python 3.12 或更高版本
- uv包管理器

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/zym9863/Time-Series-Model-Parameter-Estimator.git
cd Time-Series-Model-Parameter-Estimator
```

2. **安装依赖**
```bash
uv sync
```

3. **启动应用**
```bash
uv run streamlit run main.py
```

4. **访问应用**
打开浏览器访问 `http://localhost:8501`

## 📖 使用指南

### 数据输入方式

#### 1. 文件上传
- 支持CSV和TXT格式
- CSV文件：第一列为时间序列数据
- TXT文件：数值用空格或换行分隔

#### 2. 手动输入
- 在文本框中输入数值
- 支持逗号、空格或换行分隔
- 示例：`1.2, 2.3, 3.4, 4.5`

#### 3. 示例数据
- 使用内置的AR(2)模型生成的示例数据
- 适合快速体验和测试功能

### 模型参数设置

- **AR模型阶数**: 选择合适的阶数（建议不超过数据长度的1/4）
- 系统会自动限制最大阶数以确保计算稳定性

### 结果解读

#### 模型参数
- **AR系数 (φᵢ)**: 自回归模型的系数
- **白噪声方差 (σ²)**: 随机误差项的方差
- **模型阶数**: 所选择的AR模型阶数

#### 拟合效果
- **原始数据 vs 拟合数据**: 直观对比图表
- **残差分析**: 拟合误差的分布情况
- **局部放大**: 前50个数据点的详细对比

#### 诊断指标
- **均方误差 (MSE)**: 拟合精度指标
- **R² 决定系数**: 模型解释能力
- **AIC信息准则**: 模型选择参考
- **95%置信区间**: 预测不确定性

## 🔬 算法原理

### AR模型
自回归模型AR(p)的数学形式：
```
X(t) = φ₁X(t-1) + φ₂X(t-2) + ... + φₚX(t-p) + ε(t)
```

其中：
- `X(t)` 是时间t的观测值
- `φᵢ` 是第i个自回归系数
- `ε(t)` 是白噪声项，服从N(0, σ²)
- `p` 是模型阶数

### Yule-Walker方程
系统使用Yule-Walker方程求解AR模型参数：

1. **自协方差计算**: 计算时间序列的自协方差函数
2. **矩阵构建**: 构建Toeplitz矩阵和右侧向量
3. **线性求解**: 求解线性方程组得到AR系数
4. **方差估计**: 计算白噪声方差

## 📊 示例用法

### 示例1：使用示例数据
1. 选择"示例数据"
2. 设置AR阶数为2
3. 点击"估计模型参数"
4. 查看拟合效果和参数结果

### 示例2：手动输入数据
```
1.2, 1.5, 1.8, 2.1, 1.9, 1.6, 1.3, 1.7, 2.0, 1.8
```

### 示例3：CSV文件格式
```csv
value
1.2
1.5
1.8
2.1
1.9
```

## 🔧 开发说明

### 项目结构
```
Time-Series-Model-Parameter-Estimator/
├── main.py              # 主应用程序
├── pyproject.toml       # 项目配置
├── README.md           # 项目说明
└── .venv/              # 虚拟环境
```

### 核心类和函数
- `ARModelEstimator`: AR模型参数估计器类
- `estimate_parameters()`: 参数估计方法
- `fit_model()`: 模型拟合方法
- `load_data_from_file()`: 文件数据加载
- `parse_manual_input()`: 手动输入解析

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境设置
```bash
# 安装开发依赖
uv add --dev pytest black flake8

# 运行测试
uv run pytest

# 代码格式化
uv run black main.py
```

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🙏 致谢

感谢以下开源项目的支持：
- [Streamlit](https://streamlit.io/) - 优秀的Web应用框架
- [NumPy](https://numpy.org/) - 强大的数值计算库
- [SciPy](https://scipy.org/) - 科学计算工具包
- [Matplotlib](https://matplotlib.org/) - 数据可视化库

---

如有问题或建议，请提交Issue或联系开发者。