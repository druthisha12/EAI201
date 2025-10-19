import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# First, let's load and inspect the data
print("Loading and inspecting data...")
try:
    # Try to load the dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
except FileNotFoundError:
    print("File not found! Creating sample data for demonstration...")
    # Create sample data if file doesn't exist
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(50, 8000, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
    })
    print("Sample data created successfully!")
    print(f"Sample data shape: {df.shape}")

# Alternative simpler preprocessing approach
def simple_preprocess_telco_data(df):
    # Create a copy
    data = df.copy()
    
    print("Starting simple preprocessing...")
    print(f"Original data shape: {data.shape}")
    
    # Step 1: Drop customerID
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
        print("Dropped customerID column")
    
    # Step 2: Handle TotalCharges - if conversion fails, drop the column
    if 'TotalCharges' in data.columns:
        try:
            print("Converting TotalCharges to numeric...")
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
            print("TotalCharges converted successfully")
        except Exception as e:
            print(f"Warning: Could not convert TotalCharges to numeric. Error: {e}. Dropping column.")
            data = data.drop('TotalCharges', axis=1)
    
    # Step 3: Keep only numeric columns for simplicity
    numeric_data = data.select_dtypes(include=[np.number])
    print(f"Found {numeric_data.shape[1]} numeric columns initially")
    
    # If we lost too many columns, try to convert object columns
    if numeric_data.shape[1] < 3:
        print("Too few numeric columns. Attempting to convert object columns...")
        object_cols = data.select_dtypes(include=['object']).columns.tolist()
        print(f"Object columns to convert: {object_cols}")
        
        for col in object_cols:
            try:
                # Fill NaN values first
                data[col] = data[col].fillna('Unknown')
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
                print(f"Converted {col} to numeric")
            except Exception as e:
                print(f"Could not convert {col}. Error: {e}. Dropping it.")
                data = data.drop(col, axis=1)
        
        numeric_data = data.select_dtypes(include=[np.number])
        print(f"After conversion: {numeric_data.shape[1]} numeric columns")
    
    # Step 4: Handle missing values
    print("Handling missing values...")
    numeric_data = numeric_data.fillna(0)
    
    # Step 5: Standardize the data
    print("Standardizing data...")
    scaler = StandardScaler()
    numeric_data_standardized = pd.DataFrame(
        scaler.fit_transform(numeric_data),
        columns=numeric_data.columns,
        index=numeric_data.index
    )
    
    final_columns = numeric_data_standardized.columns.tolist()
    
    print(f"Final columns: {final_columns}")
    print(f"Final shape: {numeric_data_standardized.shape}")
    
    return numeric_data_standardized, final_columns, scaler, None

# Try the simple approach
try:
    processed_data, numerical_cols, scaler, label_encoders = simple_preprocess_telco_data(df)
    print("\nSimple preprocessing completed successfully!")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data columns: {processed_data.columns.tolist()}")
    print("\nFirst 5 rows of processed data:")
    print(processed_data.head())
    
except Exception as e:
    print(f"Error in simple preprocessing: {e}")
    import traceback
    traceback.print_exc()

# Continue with the rest of the analysis
print("\n" + "="*50)
print("STARTING PCA AND CLUSTERING ANALYSIS")
print("="*50)

# Part B: PCA Analysis
print("\nPart B: Performing PCA Analysis...")

# Prepare data for PCA
X = processed_data

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(X)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Variance explained by first 2 components: {sum(explained_variance[:2]):.3f}")
print(f"Variance explained by first 5 components: {sum(explained_variance[:5]):.3f}")

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Individual Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()
plt.show()

# PCA with 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

print(f"Variance explained by 2 PCA components: {sum(pca_2d.explained_variance_ratio_):.3f}")

# Plot 2D PCA scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6, s=20)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f} variance explained)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f} variance explained)')
plt.title('2D PCA Projection of Telecom Customers')
plt.grid(True, alpha=0.3)
plt.show()

# Part C: K-Means Clustering
print("\nPart C: Performing K-Means Clustering...")

# Determine optimal K using Elbow method and Silhouette score
X_pca = X_pca_2d  # Using 2D PCA data for clustering

# Elbow method
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Silhouette scores
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot both methods
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K')
plt.grid(True)

plt.tight_layout()
plt.show()

# Display silhouette scores
print("Silhouette Scores:")
for k, score in zip(k_range, silhouette_scores):
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Choose optimal K based on silhouette scores
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K chosen: {optimal_k}")

# Apply K-Means with optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_optimal.fit_predict(X_pca)

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                     cmap='viridis', alpha=0.7, s=30)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f} variance explained)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f} variance explained)')
plt.title(f'K-Means Clustering (K={optimal_k}) on PCA-Reduced Data')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Add cluster labels to original data
df_with_clusters = df.copy()
df_with_clusters['Cluster'] = cluster_labels

# Part D: Cluster Analysis
print("\nPart D: Analyzing Clusters...")

# Analyze cluster characteristics
# Convert Churn to numeric if it's not already
if 'Churn' in df_with_clusters.columns and df_with_clusters['Churn'].dtype == 'object':
    df_with_clusters['Churn_numeric'] = df_with_clusters['Churn'].map({'Yes': 1, 'No': 0})
else:
    df_with_clusters['Churn_numeric'] = df_with_clusters.get('Churn', 0)

cluster_analysis = df_with_clusters.groupby('Cluster').agg({
    'tenure': 'mean',
    'MonthlyCharges': 'mean', 
    'TotalCharges': 'mean',
    'Churn_numeric': 'mean',  # Churn percentage
    'customerID': 'count'  # Cluster size
}).rename(columns={'customerID': 'Count', 'Churn_numeric': 'Churn_%'})

# Convert churn percentage to actual percentage
cluster_analysis['Churn_%'] = cluster_analysis['Churn_%'] * 100

print("Cluster Characteristics:")
print(cluster_analysis)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Tenure
axes[0, 0].bar(cluster_analysis.index, cluster_analysis['tenure'])
axes[0, 0].set_title('Average Tenure by Cluster')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Tenure (months)')

# Monthly Charges
axes[0, 1].bar(cluster_analysis.index, cluster_analysis['MonthlyCharges'])
axes[0, 1].set_title('Average Monthly Charges by Cluster')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Monthly Charges ($)')

# Total Charges
axes[1, 0].bar(cluster_analysis.index, cluster_analysis['TotalCharges'])
axes[1, 0].set_title('Average Total Charges by Cluster')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Total Charges ($)')

# Churn Rate
axes[1, 1].bar(cluster_analysis.index, cluster_analysis['Churn_%'])
axes[1, 1].set_title('Churn Rate by Cluster')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.show()

# Create customer segment profiles
def describe_clusters(df_with_clusters, cluster_num):
    try:
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_num]
        
        print(f"\n=== Cluster {cluster_num} Profile ===")
        print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(df_with_clusters)*100:.1f}%)")
        print(f"Average Tenure: {cluster_data['tenure'].mean():.1f} months")
        print(f"Average Monthly Charges: ${cluster_data['MonthlyCharges'].mean():.2f}")
        print(f"Average Total Charges: ${cluster_data['TotalCharges'].mean():.2f}")
        
        # Calculate churn rate safely
        if 'Churn_numeric' in cluster_data.columns:
            churn_rate = cluster_data['Churn_numeric'].mean() * 100
        else:
            churn_rate = (cluster_data['Churn'] == 'Yes').mean() * 100
        print(f"Churn Rate: {churn_rate:.1f}%")
        
    except Exception as e:
        print(f"Error describing cluster {cluster_num}: {e}")

# Generate profiles for all clusters
print("\nCluster Profiles:")
for cluster in sorted(df_with_clusters['Cluster'].unique()):
    describe_clusters(df_with_clusters, cluster)

print("\n" + "="*50)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*50)
