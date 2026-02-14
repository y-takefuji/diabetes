import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb  # Added LightGBM
from sklearn.cluster import FeatureAgglomeration
import shap
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
try:
    data = pd.read_excel('Gestational Diabetic Dat Set.xlsx')
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create sample data for example
    data = pd.DataFrame(np.random.randint(1, 100, size=(100, 10)), 
                       columns=[f'Feature_{i}' for i in range(1, 11)])
    data['Hemoglobin'] = np.random.rand(100) * 10 + 10
    data['Class Label(GDM /Non GDM)'] = np.random.randint(0, 2, size=100)

# Handle missing values - fill NaNs with 0
if data.isnull().sum().sum() > 0:
    print(f"Found {data.isnull().sum().sum()} NaN values. Filling with 0.")
    data.fillna(0, inplace=True)

# Show dataset shape and target distribution
X = data.drop('Class Label(GDM /Non GDM)', axis=1)
y = data['Class Label(GDM /Non GDM)']
print(f"Dataset shape: {data.shape}")
print("Target distribution:")
print(y.value_counts())

# Initialize the result dictionary
results = {
    'Method': [],
    'CV5 Accuracy': [],
    'Top 5 Features': [],
    'Top 4 Features': []
}

# Function to perform cross-validation
def perform_cv(X, y, features, model=None):
    if model is None:
        model = RandomForestClassifier(random_state=42)
    
    X_selected = X[features]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
    return round(np.mean(scores), 4)

# 1. XGBoost Feature Selection
def xgb_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_filled, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.nlargest(n_features).index.tolist()

# 2. XGB-SHAP Feature Selection - Fixed version with proper array handling
def xgb_shap_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_filled, y)
    
    # Default importances as fallback
    default_importances = model.feature_importances_
    
    # Randomly select 100 samples for SHAP calculation
    np.random.seed(42)  # For reproducibility
    n_samples = min(100, X_filled.shape[0])
    sample_idx = np.random.choice(X_filled.shape[0], n_samples, replace=False)
    X_sample = X_filled.iloc[sample_idx]
    
    try:
        # Use TreeExplainer which works well with XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Process SHAP values based on their structure
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class, use positive class (index 1)
            shap_means = np.abs(shap_values[1]).mean(axis=0)
        else:
            # For single output or binary
            shap_means = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance ranking using the mean of each feature's SHAP values
        feature_ranking = [(col, float(np.mean(imp))) for col, imp in zip(X.columns, shap_means)]
        feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in feature_ranking[:n_features]]
    
    except Exception as e:
        print(f"Error in XGB-SHAP calculation: {e}, falling back to XGB importance")
        # If SHAP fails, fall back to standard XGB importance
        feature_importances = pd.Series(default_importances, index=X.columns)
        return feature_importances.nlargest(n_features).index.tolist()

# 3. Feature Agglomeration - selecting top features across all clusters
def fa_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    n_clusters = min(X_filled.shape[1] - n_features + 1, X_filled.shape[1] - 1)
    
    if n_clusters <= 0:
        return list(X_filled.columns)[:n_features]
    
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_filled)
    
    # Calculate correlation with target for all features
    feature_importances = {}
    for feature in X_filled.columns:
        importance = abs(np.corrcoef(X_filled[feature], y)[0, 1])
        feature_importances[feature] = importance
    
    # Sort by importance and select top features across all clusters
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    return [feature for feature, _ in sorted_features[:n_features]]

# 4. Highly Variable Gene Selection (purely variance-based)
def hvgs_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    variances = X_filled.var()
    return variances.nlargest(n_features).index.tolist()

# 5. Spearman Correlation
def spearman_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    correlations = []
    for feature in X_filled.columns:
        corr, _ = spearmanr(X_filled[feature], y)
        correlations.append((feature, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [feature for feature, _ in correlations[:n_features]]

# 6. LightGBM Feature Selection
def lgbm_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_filled, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.nlargest(n_features).index.tolist()

# 7. LightGBM-SHAP Feature Selection
def lgbm_shap_feature_selection(X, y, n_features):
    # Ensure no NaN values
    X_filled = X.fillna(0)
    
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_filled, y)
    
    # Default importances as fallback
    default_importances = model.feature_importances_
    
    # Randomly select 100 samples for SHAP calculation
    np.random.seed(42)  # For reproducibility
    n_samples = min(100, X_filled.shape[0])
    sample_idx = np.random.choice(X_filled.shape[0], n_samples, replace=False)
    X_sample = X_filled.iloc[sample_idx]
    
    try:
        # Use TreeExplainer which works well with LightGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Process SHAP values based on their structure
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class, use positive class (index 1)
            shap_means = np.abs(shap_values[1]).mean(axis=0)
        else:
            # For single output or binary
            shap_means = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance ranking using the mean of each feature's SHAP values
        feature_ranking = [(col, float(np.mean(imp))) for col, imp in zip(X.columns, shap_means)]
        feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in feature_ranking[:n_features]]
    
    except Exception as e:
        print(f"Error in LGBM-SHAP calculation: {e}, falling back to LGBM importance")
        # If SHAP fails, fall back to standard LGBM importance
        feature_importances = pd.Series(default_importances, index=X.columns)
        return feature_importances.nlargest(n_features).index.tolist()

# Apply all feature selection methods
methods = [
    ('XGB', xgb_feature_selection, xgb.XGBClassifier(random_state=42)),
    ('XGB-SHAP', xgb_shap_feature_selection, xgb.XGBClassifier(random_state=42)),
    ('LightGBM', lgbm_feature_selection, lgb.LGBMClassifier(random_state=42)),  # Added LightGBM
    ('LightGBM-SHAP', lgbm_shap_feature_selection, lgb.LGBMClassifier(random_state=42)),  # Added LightGBM-SHAP
    ('FA', fa_feature_selection, RandomForestClassifier(random_state=42)),
    ('HVGS', hvgs_feature_selection, RandomForestClassifier(random_state=42)),
    ('Spearman', spearman_feature_selection, RandomForestClassifier(random_state=42))
]

for method_name, selector, model in methods:
    print(f"\nProcessing {method_name}...")
    
    try:
        # Get top 5 features from full dataset
        top5_features = selector(X, y, 5)
        print(f"Top 5 features: {top5_features}")
        
        # Remove the highest ranked feature to create reduced dataset
        top_feature = top5_features[0]
        print(f"Removing top feature: {top_feature}")
        X_reduced = X.drop(top_feature, axis=1)
        
        # Re-select top 4 features from reduced dataset
        top4_features = selector(X_reduced, y, 4)
        print(f"Top 4 features from reduced dataset: {top4_features}")
        
        # Perform cross-validation on top 4 features
        if method_name in ['FA', 'HVGS', 'Spearman']:
            cv_acc = perform_cv(X.fillna(0), y, top4_features)  # Use RF for cross-validation, filling NaNs
        else:
            cv_acc = perform_cv(X.fillna(0), y, top4_features, model)  # Fill NaNs
        
        print(f"CV5 Accuracy: {cv_acc}")
        
        # Save results
        results['Method'].append(method_name)
        results['CV5 Accuracy'].append(cv_acc)
        results['Top 5 Features'].append(', '.join(top5_features))
        results['Top 4 Features'].append(', '.join(top4_features))
    
    except Exception as e:
        print(f"Error processing {method_name}: {e}")
        import traceback
        traceback.print_exc()
        
        results['Method'].append(method_name)
        results['CV5 Accuracy'].append("Error")
        results['Top 5 Features'].append("Error")
        results['Top 4 Features'].append("Error")
        continue

# Create and save results table
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)

# Save results to CSV
results_df.to_csv('result.csv', index=False)
print("Results saved to result.csv")
