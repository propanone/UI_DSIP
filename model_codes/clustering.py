# sinistre is feature is not available, the code below is useless
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_risk_subcategories(data, model, n_clusters=5):
    """
    Analyze risk subcategories and their relationship with the 'sinistre' feature.
    
    Parameters:
    data (pd.DataFrame): Original dataset containing all features and 'sinistre'
    model: Trained XGBoost model
    n_clusters (int): Number of subcategories to create
    
    Returns:
    tuple: (subcategory_assignments, subcategory_profiles, sinistre_means)
    """
    # Get feature importance from the model
    feature_importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=False)
    
    # Select top features for clustering
    top_features = feature_importance.head(6).index.tolist()
    
    # Prepare data for clustering
    cluster_data = data[top_features].copy()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    subcategories = kmeans.fit_predict(scaled_data)
    
    # Add predictions and subcategories to the data
    results_df = data.copy()
    results_df['predicted_risk'] = model.predict(data[model.feature_names_in_])
    results_df['subcategory'] = subcategories
    
    # Calculate mean sinistre by subcategory and risk prediction
    sinistre_means = results_df.groupby(['predicted_risk', 'subcategory'])['sinistre'].agg([
        'mean',
        'count',
        'std'
    ]).round(2)
    
    # Create subcategory profiles
    subcategory_profiles = []
    for i in range(n_clusters):
        cluster_data = results_df[results_df['subcategory'] == i]
        profile = {
            'subcategory': i,
            'size': len(cluster_data),
            'mean_sinistre': cluster_data['sinistre'].mean(),
            'risk_ratio': (cluster_data['predicted_risk'] == 1).mean(),
            'characteristic_features': {}
        }
        
        # Find characteristic features
        for feature in top_features:
            if feature in NUMERIC_FEATURES:
                profile['characteristic_features'][feature] = cluster_data[feature].mean()
            else:
                profile['characteristic_features'][feature] = cluster_data[feature].mode()[0]
        
        subcategory_profiles.append(profile)
    
    # Visualize subcategories
    plt.figure(figsize=(12, 6))
    
    # Plot mean sinistre by subcategory
    sns.barplot(
        data=results_df,
        x='subcategory',
        y='sinistre',
        hue='predicted_risk',
        ci='sd'
    )
    plt.title('Mean Sinistre by Subcategory and Predicted Risk')
    plt.xlabel('Subcategory')
    plt.ylabel('Mean Sinistre')
    plt.show()
    
    return results_df['subcategory'], subcategory_profiles, sinistre_means

def predict_with_subcategory(new_data, model, kmeans_model):
    """
    Predict risk and assign subcategory for new data.
    
    Parameters:
    new_data (pd.DataFrame): New data to predict
    model: Trained XGBoost model
    kmeans_model: Trained KMeans model
    
    Returns:
    tuple: (risk_prediction, subcategory)
    """
    # Make risk prediction
    risk_pred = model.predict(new_data[model.feature_names_in_])
    
    # Assign subcategory
    scaled_data = scaler.transform(new_data[top_features])
    subcategory = kmeans_model.predict(scaled_data)
    
    return risk_pred, subcategory

# Usage example:
# First, run the analysis on your training data
subcategories, profiles, sinistre_stats = analyze_risk_subcategories(data, xgb)

# Print subcategory profiles
for profile in profiles:
    print(f"\nSubcategory {profile['subcategory']}:")
    print(f"Size: {profile['size']} samples")
    print(f"Mean Sinistre: {profile['mean_sinistre']:.2f}")
    print(f"Risk Ratio: {profile['risk_ratio']:.2%}")
    print("Characteristic features:")
    for feature, value in profile['characteristic_features'].items():
        print(f"  - {feature}: {value}")

# Print sinistre statistics
print("\nSinistre Statistics by Risk and Subcategory:")
print(sinistre_stats)