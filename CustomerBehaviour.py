import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset (assuming it comes from Step 1)
file_path = "Updated_Project1.csv"  # Use the output from Step 1
df = pd.read_csv(file_path)

# ----------------------------------------------
# Step 2: Explore Customer Behavior
# ----------------------------------------------
def explore_customer_behavior(df):
    """Visualize customer trends and identify correlations between demographics and spending habits."""
    
    # 2.1 Visualize Customer Trends
    print("\nStep 2: Exploring Customer Behavior")

    # Spending Patterns: Distribution of Purchase Amounts
    if 'Purchase Amount (USD)' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Purchase Amount (USD)'], bins=30, kde=True, color='skyblue')
        plt.title("Distribution of Purchase Amounts")
        plt.xlabel("Purchase Amount (USD)")
        plt.ylabel("Frequency")
        plt.show()
        print("- Spending Patterns: Histogram shows the distribution of purchase amounts.")

    # Popular Products: Count of Purchases by Category
    if 'Category' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette='viridis')
        plt.title("Popular Product Categories")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.show()
        print("- Popular Products: Bar plot displays the most purchased product categories.")

    # Purchase Frequency: Distribution of Purchase Frequency
    if 'Frequency of Purchases' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Frequency of Purchases'], bins=20, kde=True, color='salmon')
        plt.title("Distribution of Purchase Frequency")
        plt.xlabel("Frequency of Purchases")
        plt.ylabel("Count")
        plt.show()
        print("- Purchase Frequency: Histogram shows how often customers make purchases.")

    # 2.2 Identify Correlations Between Demographics and Spending Habits
    # Age vs Purchase Amount
    if 'Age' in df.columns and 'Purchase Amount (USD)' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df['Age'], y=df['Purchase Amount (USD)'], alpha=0.5, color='green')
        plt.title("Age vs Purchase Amount")
        plt.xlabel("Age")
        plt.ylabel("Purchase Amount (USD)")
        plt.show()
        correlation_age = df['Age'].corr(df['Purchase Amount (USD)'])
        print(f"- Correlation between Age and Purchase Amount: {correlation_age:.2f}")

    # Gender vs Purchase Amount
    if 'Gender' in df.columns and 'Purchase Amount (USD)' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df['Gender'], y=df['Purchase Amount (USD)'], palette='pastel')
        plt.title("Spending Habits by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Purchase Amount (USD)")
        plt.show()
        print("- Spending by Gender: Boxplot compares spending distributions across genders.")

# Execute Step 2
if __name__ == "__main__":
    explore_customer_behavior(df)
    print("\nCustomer behavior exploration completed.")