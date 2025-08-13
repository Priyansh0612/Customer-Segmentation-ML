import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = "Project1.csv"
df = pd.read_csv(file_path)

# ----------------------------------------------
# Step 1: Understand and Preprocess the Data
# ----------------------------------------------
def preprocess_data(df, file_path):
    """Clean the data by handling missing values, duplicates, and outliers, create new features, and save the updated dataset."""
    
    # 1.1 Understand the Data
    print("Dataset Overview:")
    print(df.info())
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # 1.2 Handle Missing Values
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Calculate and display median values for numeric columns
    median_values = df[numeric_cols].median()
    print("\nMedian Values Used for Numeric Columns Imputation:")
    print(median_values)

    # Fill missing values: Numeric with median, categorical with mode
    df[numeric_cols] = df[numeric_cols].fillna(median_values)
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    print("\nMissing Values After Imputation:")
    print(df.isnull().sum())

    # 1.3 Remove Duplicates
    df_cleaned = df.drop_duplicates()
    print(f"\nDuplicates removed. Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")

    # 1.4 Detect and Remove Outliers using IQR
    Q1 = df_cleaned[numeric_cols].quantile(0.25)
    Q3 = df_cleaned[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_outliers_removed = df_cleaned[~((df_cleaned[numeric_cols] < lower_bound) | (df_cleaned[numeric_cols] > upper_bound)).any(axis=1)]
    print(f"Outliers removed. Rows after outlier removal: {len(df_outliers_removed)}")

    # Visualize data after outlier removal
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_outliers_removed[numeric_cols])
    plt.xticks(rotation=45)
    plt.title("Boxplot After Outlier Removal")
    plt.show()

    # 1.5 Create New Features (Optional)
    # Example: Total Previous Spending (Previous Purchases * Purchase Amount)
    if 'Previous Purchases' in df_outliers_removed.columns and 'Purchase Amount (USD)' in df_outliers_removed.columns:
        df_outliers_removed['Total Previous Spending'] = df_outliers_removed['Previous Purchases'] * df_outliers_removed['Purchase Amount (USD)']
        print("\nNew Feature Added: 'Total Previous Spending'")
        print("Sample of new feature:")
        print(df_outliers_removed[['Previous Purchases', 'Purchase Amount (USD)', 'Total Previous Spending']].head())

    # 1.6 Save the Cleaned Dataset
    directory = os.path.dirname(file_path)
    cleaned_file_path = os.path.join(directory, "Updated_Project1.csv")
    df_outliers_removed.to_csv(cleaned_file_path, index=False)
    print(f"\nCleaned dataset saved to: {cleaned_file_path}")

    return df_outliers_removed

# Execute Step 1
if __name__ == "__main__":
    df_cleaned = preprocess_data(df, file_path)
    print("\nPreprocessing completed. Cleaned dataset head:")
    print(df_cleaned.head())