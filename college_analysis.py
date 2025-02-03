import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]

# Read the dataset
df = pd.read_csv('College Data.csv')

# Create a directory for saving plots
import os
if not os.path.exists('analysis_plots'):
    os.makedirs('analysis_plots')

# 1. Analyze missing data
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

# Create missing data summary
missing_summary = pd.DataFrame({
    'Missing Values': missing_data,
    'Percentage Missing': missing_percent
})
missing_summary = missing_summary[missing_summary['Missing Values'] > 0].sort_values('Missing Values', ascending=False)

# Save missing data summary
missing_summary.to_csv('analysis_plots/missing_data_summary.csv')

# Visualize missing data
plt.figure(figsize=(12, 6))
plt.bar(missing_summary.index, missing_summary['Percentage Missing'])
plt.title('Percentage of Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Percentage Missing')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_plots/missing_data_visualization.png')
plt.close()

# Handle missing values
# Fill numeric columns with median
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Continue with the rest of the analysis using cleaned data
# 1. Country-wise Distribution of Students
plt.figure(figsize=(12, 6))
country_students = df.groupby('Country')['Total Students'].sum().sort_values(ascending=False).head(10)
country_students.plot(kind='bar')
plt.title('Top 10 Countries by Total Number of Students')
plt.xlabel('Country')
plt.ylabel('Total Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_plots/country_distribution.png')
plt.close()

# 2. Gender Distribution Analysis
plt.figure(figsize=(10, 6))
gender_data = df[['Male', 'Female']].sum()
plt.pie(gender_data, labels=['Male', 'Female'], autopct='%1.1f%%')
plt.title('Overall Gender Distribution')
plt.savefig('analysis_plots/gender_distribution.png')
plt.close()

# 3. Placement Rate vs CGPA
plt.figure(figsize=(10, 6))
plt.scatter(df['CGPA'], df['Placement Rate'], alpha=0.5)
plt.title('Correlation between CGPA and Placement Rate')
plt.xlabel('CGPA')
plt.ylabel('Placement Rate (%)')
plt.savefig('analysis_plots/placement_vs_cgpa.png')
plt.close()

# 4. Branch-wise Average Placement Rate
plt.figure(figsize=(12, 6))
branch_placement = df.groupby('Branch')['Placement Rate'].mean().sort_values(ascending=False)
branch_placement.plot(kind='bar')
plt.title('Average Placement Rate by Branch')
plt.xlabel('Branch')
plt.ylabel('Average Placement Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_plots/branch_placement.png')
plt.close()

# 5. Research Output Analysis
plt.figure(figsize=(10, 6))
plt.hist(df['Research Papers Published'], bins=30)
plt.title('Distribution of Research Papers Published')
plt.xlabel('Number of Research Papers')
plt.ylabel('Frequency')
plt.savefig('analysis_plots/research_distribution.png')
plt.close()

# Generate statistical summary
stats_summary = pd.DataFrame({
    'Mean Students per College': [df['Total Students'].mean()],
    'Average CGPA': [df['CGPA'].mean()],
    'Average Placement Rate': [df['Placement Rate'].mean()],
    'Average Research Papers': [df['Research Papers Published'].mean()],
    'Total Colleges': [len(df)]
})

# Save statistical summary
stats_summary.to_csv('analysis_plots/statistical_summary.csv')

# Print key findings
print("\nMissing Data Summary:")
print(missing_summary)
print("\nKey Findings:")
print(f"Total number of colleges analyzed: {len(df)}")
print(f"Average placement rate across all colleges: {df['Placement Rate'].mean():.2f}%")
print(f"Average CGPA: {df['CGPA'].mean():.2f}")
print(f"Country with most colleges: {df['Country'].value_counts().index[0]}")
print(f"Most common branch: {df['Branch'].value_counts().index[0]}")

# Add correlation analysis
correlation_matrix = df[['Total Students', 'CGPA', 'Annual Family Income', 'Research Papers Published', 'Placement Rate']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Key Metrics')
plt.tight_layout()
plt.savefig('analysis_plots/correlation_matrix.png')
plt.close()

# 1. Advanced Country Analysis
country_stats = df.groupby('Country').agg({
    'Total Students': 'sum',
    'Placement Rate': 'mean',
    'CGPA': 'mean',
    'Research Papers Published': 'mean',
    'Faculty Count': 'mean'
}).round(2)

# Sort by total students and save top 10 countries analysis
country_stats_sorted = country_stats.sort_values('Total Students', ascending=False)
country_stats_sorted.head(10).to_csv('analysis_plots/top_10_countries_analysis.csv')

# Visualize top 5 countries' metrics
top_5_countries = country_stats_sorted.head(5)
metrics = ['Placement Rate', 'CGPA', 'Research Papers Published']

fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
fig.suptitle('Top 5 Countries Performance Metrics')

for i, metric in enumerate(metrics):
    sns.barplot(data=top_5_countries.reset_index(), x='Country', y=metric, ax=axes[i])
    axes[i].set_title(f'{metric} by Country')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('analysis_plots/top_5_countries_metrics.png')
plt.close()

# 2. Branch Analysis
# Calculate branch performance metrics
branch_stats = df.groupby('Branch').agg({
    'Total Students': 'sum',
    'Placement Rate': 'mean',
    'CGPA': 'mean',
    'Research Papers Published': 'mean'
}).round(2)

branch_stats_sorted = branch_stats.sort_values('Total Students', ascending=False)
branch_stats_sorted.to_csv('analysis_plots/branch_analysis.csv')

# Visualize branch performance
plt.figure(figsize=(15, 8))
sns.scatterplot(data=df, x='CGPA', y='Placement Rate', hue='Branch', alpha=0.6)
plt.title('CGPA vs Placement Rate by Branch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('analysis_plots/branch_performance_scatter.png')
plt.close()

# 3. Income Analysis
# Create income brackets
df['Income_Bracket'] = pd.qcut(df['Annual Family Income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Analyze performance metrics by income bracket
income_analysis = df.groupby('Income_Bracket').agg({
    'CGPA': 'mean',
    'Placement Rate': 'mean',
    'Research Papers Published': 'mean'
}).round(2)

income_analysis.to_csv('analysis_plots/income_analysis.csv')

# Visualize income impact
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Income_Bracket', y='CGPA')
plt.title('CGPA Distribution by Income Bracket')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_plots/income_cgpa_distribution.png')
plt.close()

# 4. Gender Analysis by Branch
gender_branch = df.groupby('Branch').agg({
    'Male': 'sum',
    'Female': 'sum'
})
gender_branch['Female_Percentage'] = (gender_branch['Female'] / (gender_branch['Male'] + gender_branch['Female']) * 100).round(2)
gender_branch = gender_branch.sort_values('Female_Percentage', ascending=False)
gender_branch.to_csv('analysis_plots/gender_branch_analysis.csv')

# Visualize gender distribution by branch
plt.figure(figsize=(15, 8))
sns.barplot(data=gender_branch.reset_index(), x='Branch', y='Female_Percentage')
plt.title('Female Representation by Branch (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_plots/gender_branch_distribution.png')
plt.close()

# 5. Research Impact Analysis
# Calculate correlation between research papers and other metrics
research_corr = df[['Research Papers Published', 'Placement Rate', 'CGPA', 'Faculty Count']].corr()['Research Papers Published'].sort_values(ascending=False)
research_corr.to_csv('analysis_plots/research_correlations.csv')

# Visualize research impact on placement
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Research Papers Published', y='Placement Rate', scatter_kws={'alpha':0.5})
plt.title('Impact of Research Papers on Placement Rate')
plt.tight_layout()
plt.savefig('analysis_plots/research_placement_impact.png')
plt.close()

# 6. Faculty Analysis
df['Student_Faculty_Ratio'] = df['Total Students'] / df['Faculty Count']
faculty_analysis = df.groupby('Branch')['Student_Faculty_Ratio'].agg(['mean', 'min', 'max']).round(2)
faculty_analysis.to_csv('analysis_plots/faculty_analysis.csv')

# Visualize student-faculty ratio impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Student_Faculty_Ratio', y='CGPA', alpha=0.5)
plt.title('Impact of Student-Faculty Ratio on CGPA')
plt.tight_layout()
plt.savefig('analysis_plots/faculty_ratio_impact.png')
plt.close()

# Print comprehensive insights
print("\nComprehensive Analysis Results:")
print("\n1. Top Performing Country:")
top_country = country_stats_sorted.index[0]
print(f"- {top_country}")
print(f"  * Average Placement Rate: {country_stats_sorted.loc[top_country, 'Placement Rate']:.2f}%")
print(f"  * Average CGPA: {country_stats_sorted.loc[top_country, 'CGPA']:.2f}")

print("\n2. Branch with Highest Placement Rate:")
top_branch = branch_stats_sorted.sort_values('Placement Rate', ascending=False).index[0]
print(f"- {top_branch}")
print(f"  * Placement Rate: {branch_stats_sorted.loc[top_branch, 'Placement Rate']:.2f}%")

print("\n3. Income Impact:")
print("- Average CGPA by Income Bracket:")
for bracket in income_analysis.index:
    print(f"  * {bracket}: {income_analysis.loc[bracket, 'CGPA']:.2f}")

print("\n4. Gender Distribution:")
print(f"- Branch with highest female percentage: {gender_branch.index[0]} ({gender_branch['Female_Percentage'].iloc[0]:.2f}%)")

print("\n5. Research Impact:")
print(f"- Correlation with Placement Rate: {research_corr['Placement Rate']:.3f}")

print("\n6. Faculty Analysis:")
print(f"- Average Student-Faculty Ratio: {df['Student_Faculty_Ratio'].mean():.2f}")
print(f"- Best Student-Faculty Ratio: {df['Student_Faculty_Ratio'].min():.2f}")

# Advanced analyses
def perform_statistical_analysis():
    # Calculate advanced statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = pd.DataFrame({
        'Mean': df[numerical_cols].mean(),
        'Median': df[numerical_cols].median(),
        'Std': df[numerical_cols].std(),
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis()
    })
    stats_df.to_csv('analysis_plots/advanced_statistics.csv')
    
    # Create distribution plots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.histplot(df['CGPA'], kde=True, ax=axes[0,0])
    axes[0,0].set_title('CGPA Distribution')
    
    sns.histplot(df['Placement Rate'], kde=True, ax=axes[0,1])
    axes[0,1].set_title('Placement Rate Distribution')
    
    sns.histplot(df['Research Papers Published'], kde=True, ax=axes[1,0])
    axes[1,0].set_title('Research Papers Distribution')
    
    sns.histplot(df['Annual Family Income'], kde=True, ax=axes[1,1])
    axes[1,1].set_title('Family Income Distribution')
    
    plt.tight_layout()
    plt.savefig('analysis_plots/distributions.png')
    plt.close()

def perform_clustering_analysis():
    # Prepare data for clustering
    features_for_clustering = ['CGPA', 'Placement Rate', 'Research Papers Published', 'Annual Family Income']
    X = df[features_for_clustering]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_stats = df.groupby('Cluster')[features_for_clustering].mean()
    cluster_stats.to_csv('analysis_plots/cluster_analysis.csv')
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='CGPA', y='Placement Rate', hue='Cluster', palette='deep')
    plt.title('College Clusters based on Performance Metrics')
    plt.savefig('analysis_plots/clusters.png')
    plt.close()

def build_predictive_model():
    # Prepare features for placement rate prediction
    features = ['CGPA', 'Research Papers Published', 'Faculty Count', 'Total Students']
    X = df[features]
    y = df['Placement Rate']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate model performance
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.coef_
    })
    feature_importance.to_csv('analysis_plots/placement_prediction_features.csv')
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Placement Rate')
    plt.ylabel('Predicted Placement Rate')
    plt.title(f'Placement Rate Prediction (R² = {r2:.3f}, RMSE = {rmse:.3f})')
    plt.savefig('analysis_plots/placement_prediction.png')
    plt.close()
    
    return r2, rmse

def analyze_gender_trends():
    # Calculate gender ratio trends across different metrics
    gender_metrics = pd.DataFrame({
        'Branch': df['Branch'],
        'Gender_Ratio': df['Female'] / (df['Male'] + df['Female']),
        'CGPA': df['CGPA'],
        'Placement_Rate': df['Placement Rate']
    })
    
    # Analyze gender performance by branch
    gender_performance = gender_metrics.groupby('Branch').agg({
        'Gender_Ratio': 'mean',
        'CGPA': 'mean',
        'Placement_Rate': 'mean'
    }).round(3)
    
    gender_performance.to_csv('analysis_plots/gender_performance.csv')
    
    # Visualize gender ratio vs performance
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=gender_metrics, x='Gender_Ratio', y='CGPA', hue='Branch')
    plt.title('Gender Ratio vs Academic Performance')
    plt.savefig('analysis_plots/gender_performance_scatter.png')
    plt.close()

def analyze_geographic_patterns():
    # Calculate regional statistics
    regional_stats = df.groupby('Country').agg({
        'Total Students': 'sum',
        'Placement Rate': 'mean',
        'CGPA': 'mean',
        'Research Papers Published': 'sum',
        'Faculty Count': 'sum'
    }).round(2)
    
    # Calculate student-faculty ratio by country
    regional_stats['Student_Faculty_Ratio'] = (
        regional_stats['Total Students'] / regional_stats['Faculty Count']
    ).round(2)
    
    regional_stats.to_csv('analysis_plots/regional_analysis.csv')
    
    # Create geographic performance visualization
    plt.figure(figsize=(15, 8))
    sns.barplot(data=regional_stats.reset_index().nlargest(10, 'Placement Rate'), 
                x='Country', y='Placement Rate')
    plt.title('Top 10 Countries by Placement Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_plots/geographic_performance.png')
    plt.close()

# Run all analyses
print("Running comprehensive college data analysis...")

print("\n1. Performing Statistical Analysis...")
perform_statistical_analysis()

print("\n2. Performing Clustering Analysis...")
perform_clustering_analysis()

print("\n3. Building Predictive Model...")
r2, rmse = build_predictive_model()
print(f"Placement Rate Prediction Results:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

print("\n4. Analyzing Gender Trends...")
analyze_gender_trends()

print("\n5. Analyzing Geographic Patterns...")
analyze_geographic_patterns()

# Print comprehensive insights
print("\nKey Insights Generated:")
print("1. Statistical Analysis: Check 'advanced_statistics.csv' for detailed statistics")
print("2. Clustering Analysis: Colleges grouped into 4 clusters based on performance metrics")
print("3. Predictive Modeling: Model created to predict placement rates")
print("4. Gender Analysis: Detailed gender performance metrics by branch available")
print("5. Geographic Analysis: Regional patterns in educational outcomes analyzed")

print("\nAll visualizations and data files have been saved in the 'analysis_plots' directory.")
