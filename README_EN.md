# Time Series Model Parameter Estimator

A Python and Streamlit-based tool for estimating AR model parameters, offering an intuitive web interface and powerful analysis features.

[中文文档 (Chinese)](README.md)

## 🚀 Features

### Core Features
- **AR Model Parameter Estimation**: Accurately calculates AR model coefficients and white noise variance using the Yule-Walker equations
- **Model Fitting & Visualization**: Generates fitted values and visually compares them with the original data
- **Multiple Data Input Methods**: Supports file upload, manual input, and sample data
- **Model Diagnostics**: Provides residual analysis, goodness-of-fit, and other professional diagnostic metrics

### UI Highlights
- 🎨 Modern web interface design
- 📊 Interactive chart display
- 📈 Real-time parameter calculation
- 🔍 Zoom-in for local fitting effect
- 📋 Detailed model summary report

## 🛠️ Tech Stack

- **Python 3.12+**: Core programming language
- **Streamlit**: Web UI framework
- **NumPy**: Numerical computation
- **Pandas**: Data processing
- **Matplotlib**: Data visualization
- **SciPy**: Scientific computing
- **uv**: Modern Python package manager

## 📦 Installation & Running

### Requirements
- Python 3.12 or higher
- uv package manager

### Quick Start

1. **Clone the project**
```bash
git clone https://github.com/zym9863/Time-Series-Model-Parameter-Estimator.git
cd Time-Series-Model-Parameter-Estimator
```

2. **Install dependencies**
```bash
uv sync
```

3. **Start the app**
```bash
uv run streamlit run main.py
```

4. **Access the app**
Open your browser and visit `http://localhost:8501`

## 📖 User Guide

### Data Input Methods

#### 1. File Upload
- Supports CSV and TXT formats
- CSV: First column is the time series data
- TXT: Values separated by spaces or newlines

#### 2. Manual Input
- Enter values in the text box
- Supports comma, space, or newline separation
- Example: `1.2, 2.3, 3.4, 4.5`

#### 3. Sample Data
- Uses built-in AR(2) model sample data
- Ideal for quick experience and testing

### Model Parameter Settings

- **AR Model Order**: Choose an appropriate order (recommended not to exceed 1/4 of data length)
- The system automatically limits the maximum order for stability

### Result Interpretation

#### Model Parameters
- **AR Coefficients (φᵢ)**: Coefficients of the autoregressive model
- **White Noise Variance (σ²)**: Variance of the random error term
- **Model Order**: The selected AR model order

#### Fitting Effect
- **Original vs Fitted Data**: Intuitive comparison chart
- **Residual Analysis**: Distribution of fitting errors
- **Zoom-in**: Detailed comparison of the first 50 data points

#### Diagnostic Metrics
- **Mean Squared Error (MSE)**: Fitting accuracy metric
- **R² Coefficient of Determination**: Model explanatory power
- **AIC Information Criterion**: Model selection reference
- **95% Confidence Interval**: Prediction uncertainty

## 🔬 Algorithm Principle

### AR Model
The mathematical form of the AR(p) model:
```
X(t) = φ₁X(t-1) + φ₂X(t-2) + ... + φₚX(t-p) + ε(t)
```
Where:
- `X(t)`: Observation at time t
- `φᵢ`: The i-th autoregressive coefficient
- `ε(t)`: White noise term, follows N(0, σ²)
- `p`: Model order

### Yule-Walker Equations
The system uses the Yule-Walker equations to solve AR model parameters:

1. **Autocovariance Calculation**: Compute the autocovariance function of the time series
2. **Matrix Construction**: Build the Toeplitz matrix and right-hand vector
3. **Linear Solution**: Solve the linear system for AR coefficients
4. **Variance Estimation**: Calculate white noise variance

## 📊 Example Usage

### Example 1: Using Sample Data
1. Select "Sample Data"
2. Set AR order to 2
3. Click "Estimate Model Parameters"
4. View fitting effect and parameter results

### Example 2: Manual Data Input
```
1.2, 1.5, 1.8, 2.1, 1.9, 1.6, 1.3, 1.7, 2.0, 1.8
```

### Example 3: CSV File Format
```csv
value
1.2
1.5
1.8
2.1
1.9
```

## 🔧 Development Notes

### Project Structure
```
Time-Series-Model-Parameter-Estimator/
├── main.py              # Main application
├── pyproject.toml       # Project config
├── README.md           # Project description
└── .venv/              # Virtual environment
```

### Core Classes & Functions
- `ARModelEstimator`: AR model parameter estimator class
- `estimate_parameters()`: Parameter estimation method
- `fit_model()`: Model fitting method
- `load_data_from_file()`: File data loader
- `parse_manual_input()`: Manual input parser

## 🤝 Contributing

Contributions via Issues and Pull Requests are welcome!

### Development Setup
```bash
# Install dev dependencies
uv add --dev pytest black flake8

# Run tests
uv run pytest

# Code formatting
uv run black main.py
```

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

## 🙏 Acknowledgements

Thanks to the following open-source projects:
- [Streamlit](https://streamlit.io/) - Excellent web app framework
- [NumPy](https://numpy.org/) - Powerful numerical library
- [SciPy](https://scipy.org/) - Scientific computing toolkit
- [Matplotlib](https://matplotlib.org/) - Data visualization library

---

For questions or suggestions, please open an Issue or contact the developer.
