import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_ml_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="College Analysis Dashboard", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Define numeric columns
    numeric_cols = ['CGPA', 'Research Papers Published', 'Faculty Count', 
                   'Total Students', 'Female', 'Male', 'Placement Rate',
                   'Annual Family Income']
    
    # Load data with specific data types
    df = pd.read_csv('College Data.csv')
    
    # Convert numeric columns to float
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def main():
    # Title and introduction
    st.title("🎓 College Data Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of college statistics using machine learning.
    Explore different aspects of the data through various analyses and visualizations.
    """)

    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis",
        ["Overview", "Data Analysis", "Feature Engineering", "ML Models", "Advanced Analysis"]
    )

    if analysis_type == "Overview":
        show_overview(df)
    elif analysis_type == "Data Analysis":
        show_data_analysis(df)
    elif analysis_type == "Feature Engineering":
        show_feature_engineering(df)
    elif analysis_type == "ML Models":
        show_ml_models(df)
    else:
        show_advanced_analysis(df)

def show_overview(df):
    st.header("Dataset Overview")
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Colleges", len(df))
    with col2:
        st.metric("Average CGPA", f"{df['CGPA'].mean():.2f}")
    with col3:
        st.metric("Average Placement Rate", f"{df['Placement Rate'].mean():.2f}%")

    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Show correlation heatmap for numeric columns only
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig = px.imshow(df[numeric_cols].corr(), 
                    color_continuous_scale='RdBu',
                    title="Feature Correlations")
    st.plotly_chart(fig)

def show_data_analysis(df):
    st.header("Exploratory Data Analysis")

    # Distribution plots
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select Feature for Distribution", 
                             df.select_dtypes(include=[np.number]).columns)
        fig = px.histogram(df, x=feature, nbins=30, 
                          title=f"Distribution of {feature}")
        st.plotly_chart(fig)

    with col2:
        scatter_x = st.selectbox("Select X-axis feature", 
                               df.select_dtypes(include=[np.number]).columns,
                               index=0)
        scatter_y = st.selectbox("Select Y-axis feature", 
                               df.select_dtypes(include=[np.number]).columns,
                               index=1)
        fig = px.scatter(df, x=scatter_x, y=scatter_y, 
                        title=f"{scatter_x} vs {scatter_y}")
        st.plotly_chart(fig)

    # Country-wise analysis
    st.subheader("Country-wise Analysis")
    country_stats = df.groupby('Country').agg({
        'Placement Rate': 'mean',
        'CGPA': 'mean',
        'Total Students': 'sum'
    }).round(2)
    
    metric = st.selectbox("Select Metric", ['Placement Rate', 'CGPA', 'Total Students'])
    fig = px.bar(country_stats.nlargest(10, metric), 
                 y=metric, 
                 title=f"Top 10 Countries by {metric}")
    st.plotly_chart(fig)

def show_feature_engineering(df):
    st.header("Feature Engineering")

    # Create advanced features
    df_engineered = create_advanced_features(df.copy())
    
    # Show new features
    st.subheader("Engineered Features")
    new_features = ['Student_Faculty_Ratio', 'Female_Ratio', 'Research_Per_Faculty',
                   'CGPA_Research', 'Income_Faculty_Ratio', 'CGPA_Squared', 'Research_Squared']
    
    st.dataframe(df_engineered[new_features].head())

    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    X = df_engineered[new_features]
    y = df_engineered['Placement Rate']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': new_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 orientation='h',
                 title="Feature Importance")
    st.plotly_chart(fig)

def show_ml_models(df):
    st.header("Machine Learning Models")

    # Prepare data
    df_engineered = create_advanced_features(df.copy())
    features = ['CGPA', 'Research Papers Published', 'Faculty Count', 'Total Students',
               'Student_Faculty_Ratio', 'Female_Ratio', 'Research_Per_Faculty']
    
    X = df_engineered[features]
    y = df_engineered['Placement Rate']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "XGBoost", "Ensemble"]
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                model = create_ensemble_model(X_train_scaled, y_train)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
            with col3:
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")

            # Plot actual vs predicted
            fig = px.scatter(x=y_test, y=y_pred, 
                           labels={'x': 'Actual', 'y': 'Predicted'},
                           title="Actual vs Predicted Values")
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction'))
            st.plotly_chart(fig)

            # Show residual analysis
            st.subheader("Residual Analysis")
            analyze_residuals(y_test, y_pred)

def show_advanced_analysis(df):
    st.header("Advanced Analysis")

    # Clustering analysis
    st.subheader("Clustering Analysis")
    features_for_clustering = ['CGPA', 'Placement Rate', 'Research Papers Published']
    n_clusters = st.slider("Number of Clusters", 2, 8, 4)
    
    if st.button("Perform Clustering"):
        clusters, centers = perform_advanced_clustering(df[features_for_clustering], n_clusters)
        df['Cluster'] = clusters
        
        fig = px.scatter_3d(df, x='CGPA', y='Placement Rate', 
                           z='Research Papers Published',
                           color='Cluster',
                           title="3D Cluster Visualization")
        st.plotly_chart(fig)

    # Feature interaction analysis
    st.subheader("Feature Interaction Analysis")
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select First Feature", df.select_dtypes(include=[np.number]).columns)
    with col2:
        feature2 = st.selectbox("Select Second Feature", df.select_dtypes(include=[np.number]).columns)
    
    if st.button("Analyze Interactions"):
        analyze_feature_interactions(df, feature1, feature2, 'Placement Rate')

if __name__ == "__main__":
    main()
