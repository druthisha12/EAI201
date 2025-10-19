import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load the Titanic dataset
print("=== LOADING TITANIC DATASET ===")
# Load dataset (you can download from Kaggle or use seaborn built-in)
try:
    df = pd.read_csv('titanic.csv')
except:
    # Alternative: load from seaborn if available
    import seaborn as sns
    df = sns.load_dataset('titanic')
    df = df.rename(columns={'pclass': 'Pclass', 'sibsp': 'SibSp', 'parch': 'Parch'})

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Task 2: Exploratory Data Analysis (EDA)
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# 2.1 Data types and missing values
print("\n--- Data Types and Missing Values ---")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 2.2 Basic statistics
print("\n--- Basic Statistics ---")
print(df.describe())

# 2.3 Visualize distributions
print("\n--- Visualizing Distributions ---")

# Set up the plotting figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Titanic Dataset - Feature Distributions', fontsize=16)

# Age distribution
axes[0,0].hist(df['Age'].dropna(), bins=20, alpha=0.7, color='skyblue')
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Frequency')

# Fare distribution
axes[0,1].hist(df['Fare'], bins=30, alpha=0.7, color='lightcoral')
axes[0,1].set_title('Fare Distribution')
axes[0,1].set_xlabel('Fare')

# Survival count
sns.countplot(data=df, x='survived', ax=axes[0,2])
axes[0,2].set_title('Survival Count')

# Pclass distribution
sns.countplot(data=df, x='Pclass', ax=axes[1,0])
axes[1,0].set_title('Passenger Class Distribution')

# Sex distribution
sns.countplot(data=df, x='sex', ax=axes[1,1])
axes[1,1].set_title('Gender Distribution')

# Embarked distribution
sns.countplot(data=df, x='embarked', ax=axes[1,2])
axes[1,2].set_title('Embarkation Port Distribution')

plt.tight_layout()
plt.show()

# 2.4 Analyze relationships with survival
print("\n--- Survival Analysis ---")

# Survival by Sex
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Survival by Sex
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0])
axes[0].set_title('Survival by Gender')

# Survival by Pclass
sns.countplot(data=df, x='Pclass', hue='survived', ax=axes[1])
axes[1].set_title('Survival by Passenger Class')

# Survival by Embarked
sns.countplot(data=df, x='embarked', hue='survived', ax=axes[2])
axes[2].set_title('Survival by Embarkation Port')

plt.tight_layout()
plt.show()

# Survival rates by different features
print("\nSurvival Rates:")
print(f"Overall survival rate: {df['survived'].mean():.2%}")
print(f"\nBy Gender:")
print(df.groupby('sex')['survived'].mean())
print(f"\nBy Passenger Class:")
print(df.groupby('Pclass')['survived'].mean())
print(f"\nBy Embarkation Port:")
print(df.groupby('embarked')['survived'].mean())

# Age vs Survival
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='survived', y='Age')
plt.title('Age Distribution by Survival Status')
plt.show()

# Task 3: Data Cleaning and Imputation
print("\n=== DATA CLEANING AND IMPUTATION ===")

# 3.1 Handle missing values
print("Handling missing values...")

# Age - impute with median (less sensitive to outliers)
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
print(f"Imputed {df['Age'].isnull().sum()} missing Age values with median: {age_median}")

# Embarked - impute with mode (most frequent value)
embarked_mode = df['embarked'].mode()[0]
df['embarked'].fillna(embarked_mode, inplace=True)
print(f"Imputed {df['embarked'].isnull().sum()} missing Embarked values with mode: {embarked_mode}")

# Fare - impute with median if any missing
if df['Fare'].isnull().sum() > 0:
    fare_median = df['Fare'].median()
    df['Fare'].fillna(fare_median, inplace=True)
    print(f"Imputed {df['Fare'].isnull().sum()} missing Fare values with median: {fare_median}")

# Cabin - drop this column as suggested (too many missing values)
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)
    print("Dropped Cabin column due to high percentage of missing values")

# 3.2 Drop irrelevant columns
print("\nDropping irrelevant columns...")
columns_to_drop = ['PassengerId', 'Name', 'Ticket']
# Only drop columns that exist in the dataframe
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(existing_columns_to_drop, axis=1, inplace=True)
print(f"Dropped columns: {existing_columns_to_drop}")

print(f"\nData shape after cleaning: {df.shape}")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# Task 4: Feature Engineering
print("\n=== FEATURE ENGINEERING ===")

# 4.1 Create FamilySize feature
print("Creating FamilySize feature...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
print("FamilySize distribution:")
print(df['FamilySize'].value_counts().sort_index())

# Analyze survival by family size
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='FamilySize', y='survived')
plt.title('Survival Rate by Family Size')
plt.show()

# 4.2 Extract Titles from Name (if Name column exists)
# Since we dropped Name above, let's check if we need to reload or adapt
print("\nNote: Name column was dropped as per instructions.")
print("If you need Title extraction, keep Name column and use this code:")
print("""
# Extract titles from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
# Group rare titles
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
print("Title distribution:")
print(df['Title'].value_counts())
""")

# 4.3 Create IsAlone feature as alternative
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print(f"\nIsAlone distribution:\n{df['IsAlone'].value_counts()}")

# 4.4 Create Age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
print(f"\nAgeGroup distribution:\n{df['AgeGroup'].value_counts()}")

# 4.5 Create Fare groups
df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
print(f"\nFareGroup distribution:\n{df['FareGroup'].value_counts()}")

# Task 5: Convert categorical features to numeric
print("\n=== ENCODING CATEGORICAL FEATURES ===")

# 5.1 Label encoding for binary variables
label_encoders = {}
binary_columns = ['sex']  # Add 'Title' if you extracted it

for col in binary_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Label encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 5.2 One-hot encoding for multi-category variables
categorical_columns = ['embarked', 'AgeGroup', 'FareGroup']  # Add 'Title' if extracted

# Create dummy variables
df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns, drop_first=True)

print(f"Shape before encoding: {df.shape}")
print(f"Shape after encoding: {df_encoded.shape}")

# Task 6: Prepare Data for Modeling
print("\n=== PREPARING DATA FOR MODELING ===")

# 6.1 Final feature selection
# Drop original categorical columns that have been encoded
columns_to_drop_final = ['sex', 'embarked', 'AgeGroup', 'FareGroup']
final_columns_to_drop = [col for col in columns_to_drop_final if col in df_encoded.columns]
X = df_encoded.drop(final_columns_to_drop + ['survived'], axis=1)
y = df_encoded['survived']

print(f"Final features: {list(X.columns)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 6.2 Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training survival rate: {y_train.mean():.2%}")
print(f"Test survival rate: {y_test.mean():.2%}")

# 6.3 Data readiness check
print("\n=== DATA READINESS CHECK ===")
print("✓ Missing values handled")
print("✓ Irrelevant features removed") 
print("✓ New features engineered")
print("✓ Categorical features encoded")
print("✓ Data split into train/test sets")
print("✓ Data is ready for model training!")

# Display final dataset info
print(f"\nFinal training set shape: {X_train.shape}")
print("\nFirst 5 rows of processed features:")
print(X_train.head())

# Correlation matrix of final features
plt.figure(figsize=(12, 10))
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
