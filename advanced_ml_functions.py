import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """Create advanced features for analysis"""
    # Basic ratios
    df['Student_Faculty_Ratio'] = df['Total Students'] / df['Faculty Count']
    df['Female_Ratio'] = df['Female'] / df['Total Students']
    df['Research_Per_Faculty'] = df['Research Papers Published'] / df['Faculty Count']
    
    # Interaction features
    df['CGPA_Research'] = df['CGPA'] * df['Research Papers Published']
    df['Income_Faculty_Ratio'] = df['Annual Family Income'] / df['Faculty Count']
    
    # Polynomial features for key metrics
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['CGPA', 'Research Papers Published']])
    df['CGPA_Squared'] = poly_features[:, -2]
    df['Research_Squared'] = poly_features[:, -1]
    
    return df

def perform_advanced_clustering(X, n_clusters=4):
    """Perform advanced clustering analysis"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('College Clusters (PCA Visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.show()
    
    return clusters, kmeans.cluster_centers_

def create_ensemble_model(X_train, y_train):
    """Create an ensemble model combining multiple regressors"""
    # Base models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    # Create voting regressor
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb_model)
    ])
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    return ensemble

def plot_learning_curves(estimator, X, y, title):
    """Plot learning curves for a model"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def analyze_feature_interactions(df, feature1, feature2, target):
    """Analyze and visualize feature interactions"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue=target, size=target,
                    sizes=(20, 200), palette='viridis')
    plt.title(f'Interaction between {feature1} and {feature2}')
    plt.show()
    
    # Calculate correlation
    correlation = df[[feature1, feature2, target]].corr()
    print(f"\nCorrelation Matrix:")
    print(correlation)

def perform_statistical_tests(df, feature, target):
    """Perform statistical tests on features"""
    # Pearson correlation
    correlation, p_value = stats.pearsonr(df[feature], df[target])
    print(f"\nPearson Correlation ({feature} vs {target}):")
    print(f"Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Split feature into quartiles
    quartiles = pd.qcut(df[feature], q=4)
    quartile_means = df.groupby(quartiles)[target].mean()
    
    # ANOVA test
    groups = [group for name, group in df.groupby(quartiles)[target]]
    f_statistic, anova_p_value = stats.f_oneway(*groups)
    
    print(f"\nANOVA Test Results:")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {anova_p_value:.4f}")
    
    # Visualize quartile analysis
    plt.figure(figsize=(10, 6))
    quartile_means.plot(kind='bar')
    plt.title(f'Mean {target} by {feature} Quartiles')
    plt.xlabel('Quartiles')
    plt.ylabel(f'Mean {target}')
    plt.xticks(rotation=45)
    plt.show()

def create_prediction_intervals(model, X_test, y_test, confidence=0.95):
    """Create prediction intervals using bootstrap"""
    predictions = []
    n_bootstraps = 1000
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.randint(0, len(X_test), len(X_test))
        sample_pred = model.predict(X_test[indices])
        predictions.append(sample_pred)
    
    predictions = np.array(predictions)
    
    # Calculate prediction intervals
    lower = np.percentile(predictions, ((1 - confidence) / 2) * 100, axis=0)
    upper = np.percentile(predictions, (1 - (1 - confidence) / 2) * 100, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, mean_pred, alpha=0.5)
    plt.fill_between(y_test, lower, upper, alpha=0.2, label=f'{confidence*100}% Prediction Interval')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions with Confidence Intervals')
    plt.legend()
    plt.show()

def analyze_residuals(y_true, y_pred):
    """Analyze model residuals"""
    residuals = y_true - y_pred
    
    # Create residual plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    axes[0,0].scatter(y_pred, residuals, alpha=0.5)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_xlabel('Predicted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted Values')
    
    # Residual distribution
    axes[0,1].hist(residuals, bins=30, edgecolor='black')
    axes[0,1].set_title('Residual Distribution')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot')
    
    # Residual scatter
    axes[1,1].scatter(range(len(residuals)), residuals, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Index')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residual Scatter')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate residual statistics
    print("\nResidual Statistics:")
    print(f"Mean of Residuals: {np.mean(residuals):.4f}")
    print(f"Std of Residuals: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    # Perform normality test
    _, normality_p_value = stats.normaltest(residuals)
    print(f"Normality Test p-value: {normality_p_value:.4f}")
