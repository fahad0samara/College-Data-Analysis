# College Data Analysis Dashboard 🎓

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An advanced analytics dashboard for analyzing college statistics using machine learning, providing interactive visualizations and predictive insights.

## 📊 Dashboard Preview

The dashboard includes multiple sections for comprehensive analysis:

### 1. Overview Section
- Dataset statistics and summary
- Interactive correlation heatmaps
- Basic metrics visualization

### 2. Data Analysis Section
- Distribution analysis
- Feature relationships
- Geographic patterns
- Gender distribution analysis

### 3. Machine Learning Section
- Predictive modeling
- Model comparison
- Feature importance
- Performance metrics

### 4. Advanced Analytics
- Clustering analysis
- Statistical testing
- Custom feature engineering
- Prediction intervals

## 🎯 Key Features

### Data Analysis
- **Comprehensive Statistics**: Analyze key metrics like CGPA, placement rates, and research output
- **Interactive Visualizations**: Dynamic plots and charts for data exploration
- **Correlation Analysis**: Understand relationships between different variables
- **Geographic Insights**: Country-wise performance comparison

### Machine Learning Models
- **Multiple Algorithms**:
  - Random Forest Regression
  - Gradient Boosting
  - XGBoost
  - Ensemble Models
- **Model Evaluation**:
  - R² Score
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Cross-validation scores

### Advanced Features
- **Feature Engineering**:
  - Student-Faculty Ratio
  - Research per Faculty
  - Gender Distribution Metrics
- **Clustering Analysis**:
  - K-means clustering
  - PCA visualization
  - Cluster interpretation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/fahad0samara/College-Data-Analysis.git
   cd College-Data-Analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python -m streamlit run college_analysis_app.py
   ```

## 💡 Usage Guide

### 1. Data Overview
- View basic statistics and dataset information
- Explore correlations between variables
- Analyze distribution patterns

### 2. Feature Analysis
- Select specific features for detailed analysis
- Compare different metrics
- Visualize relationships

### 3. ML Model Training
- Choose a machine learning model
- Train on selected features
- View performance metrics
- Analyze predictions

### 4. Advanced Analysis
- Perform clustering analysis
- View feature interactions
- Generate statistical reports

## 📁 Project Structure

```
College-Data-Analysis/
├── 📊 analysis_plots/        # Generated visualizations
├── 📈 ml_results/           # ML model outputs
├── 📝 college_analysis_app.py   # Main Streamlit app
├── 🔧 advanced_ml_functions.py  # ML utilities
├── 📋 requirements.txt      # Dependencies
└── 📖 README.md            # Documentation
```

## 🛠️ Technologies Used

- **Frontend**: 
  - Streamlit (Interactive Dashboard)
  - Plotly (Interactive Plots)
  - Matplotlib & Seaborn (Visualizations)

- **Backend**:
  - Python 3.8+
  - Pandas (Data Processing)
  - NumPy (Numerical Operations)

- **Machine Learning**:
  - scikit-learn
  - XGBoost
  - Feature-engine

## 📊 Sample Visualizations

The dashboard includes various types of visualizations:
- Correlation Heatmaps
- Distribution Plots
- Scatter Plots
- Bar Charts
- 3D Cluster Visualizations
- Feature Importance Plots

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Fahad Samara**
- GitHub: [@fahad0samara](https://github.com/fahad0samara)
- LinkedIn: [Fahad Samara](https://linkedin.com/in/fahad-samara)

## 🌟 Acknowledgments

- Dataset source: Global College Statistics
- Streamlit community
- scikit-learn documentation
- Python data science community
