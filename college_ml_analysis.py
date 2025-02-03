import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]

# Create directory for ML results
import os
if not os.path.exists('ml_results'):
    os.makedirs('ml_results')

# Load the data
print("Loading and preparing data...")
df = pd.read_csv('College Data.csv')

class CollegePredictor:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        
    def prepare_data(self):
        # Select features for prediction
        self.features = ['CGPA', 'Research Papers Published', 'Faculty Count', 
                        'Total Students', 'Male', 'Female', 'Annual Family Income']
        self.target = 'Placement Rate'
        
        # Prepare X and y
        X = df[self.features]
        y = df[self.target]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Save the scaler
        joblib.dump(scaler, 'ml_results/scaler.joblib')
        
    def train_models(self):
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Store model and predictions
            self.models[name] = model
            self.predictions[name] = {
                'train': train_pred,
                'test': test_pred
            }
            
            # Calculate and store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
            
            # Save model
            joblib.dump(model, f'ml_results/{name.lower().replace(" ", "_")}_model.joblib')
            
    def evaluate_models(self):
        # Create results dictionary
        results = {
            'Model': [],
            'Train R2': [],
            'Test R2': [],
            'Train RMSE': [],
            'Test RMSE': [],
            'Train MAE': [],
            'Test MAE': []
        }
        
        # Evaluate each model
        for name in self.models.keys():
            train_pred = self.predictions[name]['train']
            test_pred = self.predictions[name]['test']
            
            # Calculate metrics
            results['Model'].append(name)
            results['Train R2'].append(r2_score(self.y_train, train_pred))
            results['Test R2'].append(r2_score(self.y_test, test_pred))
            results['Train RMSE'].append(np.sqrt(mean_squared_error(self.y_train, train_pred)))
            results['Test RMSE'].append(np.sqrt(mean_squared_error(self.y_test, test_pred)))
            results['Train MAE'].append(mean_absolute_error(self.y_train, train_pred))
            results['Test MAE'].append(mean_absolute_error(self.y_test, test_pred))
        
        # Create and save results DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv('ml_results/model_comparison.csv', index=False)
        return results_df
    
    def plot_results(self):
        # 1. Model Comparison Plot
        results_df = pd.read_csv('ml_results/model_comparison.csv')
        
        plt.figure(figsize=(15, 6))
        x = np.arange(len(results_df['Model']))
        width = 0.35
        
        plt.bar(x - width/2, results_df['Test R2'], width, label='Test R²')
        plt.bar(x + width/2, results_df['Train R2'], width, label='Train R²')
        
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, results_df['Model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('ml_results/model_comparison.png')
        plt.close()
        
        # 2. Feature Importance Plots
        for name, importance_df in self.feature_importance.items():
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title(f'Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(f'ml_results/feature_importance_{name.lower().replace(" ", "_")}.png')
            plt.close()
        
        # 3. Prediction vs Actual Plots
        best_model_name = results_df.loc[results_df['Test R2'].idxmax(), 'Model']
        best_predictions = self.predictions[best_model_name]['test']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, best_predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Placement Rate')
        plt.ylabel('Predicted Placement Rate')
        plt.title(f'Best Model ({best_model_name}) Predictions vs Actual')
        plt.tight_layout()
        plt.savefig('ml_results/best_model_predictions.png')
        plt.close()

    def tune_best_model(self):
        # Get the best model based on test R2 score
        results_df = pd.read_csv('ml_results/model_comparison.csv')
        best_model_name = results_df.loc[results_df['Test R2'].idxmax(), 'Model']
        
        print(f"\nTuning {best_model_name}...")
        
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42)
            
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        elif best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5]
            }
            base_model = xgb.XGBRegressor(random_state=42)
        
        else:
            print("Tuning not implemented for this model type")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Save best parameters and score
        best_params = pd.DataFrame([grid_search.best_params_])
        best_params.to_csv('ml_results/best_model_params.csv', index=False)
        
        # Train final model with best parameters
        final_model = grid_search.best_estimator_
        final_model.fit(self.X_train_scaled, self.y_train)
        
        # Save final model
        joblib.dump(final_model, 'ml_results/best_tuned_model.joblib')
        
        # Evaluate final model
        train_pred = final_model.predict(self.X_train_scaled)
        test_pred = final_model.predict(self.X_test_scaled)
        
        final_results = {
            'Model': ['Tuned ' + best_model_name],
            'Train R2': [r2_score(self.y_train, train_pred)],
            'Test R2': [r2_score(self.y_test, test_pred)],
            'Train RMSE': [np.sqrt(mean_squared_error(self.y_train, train_pred))],
            'Test RMSE': [np.sqrt(mean_squared_error(self.y_test, test_pred))],
            'Train MAE': [mean_absolute_error(self.y_train, train_pred)],
            'Test MAE': [mean_absolute_error(self.y_test, test_pred)]
        }
        
        pd.DataFrame(final_results).to_csv('ml_results/tuned_model_results.csv', index=False)

# Run the analysis
print("Starting Machine Learning Analysis...")
predictor = CollegePredictor()

print("\nPreparing data...")
predictor.prepare_data()

print("\nTraining models...")
predictor.train_models()

print("\nEvaluating models...")
results = predictor.evaluate_models()
print("\nModel Comparison Results:")
print(results)

print("\nCreating visualizations...")
predictor.plot_results()

print("\nTuning best model...")
predictor.tune_best_model()

print("\nAnalysis complete! Check the 'ml_results' directory for detailed results and visualizations.")
