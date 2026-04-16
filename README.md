<h1 align="center">
  Customer Segmentation ML Project
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib">
</p>

## 📌 Overview

**Customer Segmentation ML** is a data science project that groups retail customers based on their behavior and purchasing patterns. By employing various machine learning clustering techniques—such as K-Means and Agglomerative Clustering—this project derives actionable insights that can drive targeted marketing strategies and improve customer retention.

## 📂 Project Structure

This project follows an industry-standard directory structure to ensure maintainability and a clear separation of concerns.

```text
Customer-Segmentation-ML/
├── data/
│   ├── raw/                # Original, raw datasets (Project1.csv)
│   └── processed/          # Cleaned dataset (Updated_Project1.csv)
├── src/
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── customer_behavior.py# Exploratory Data Analysis (EDA)
│   └── clustering.py       # Machine learning clustering algorithms
├── reports/
│   ├── Project_1_Report.pdf 
│   └── Project_1_Presentation.pptx
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Installation

Clone the repository and install the required packages using `pip`.

```bash
git clone https://github.com/Priyansh0612/Customer-Segmentation-ML.git
cd Customer-Segmentation-ML
pip install -r requirements.txt
```

### 3. Usage

To run the pipeline from start to finish, execute the scripts located in the `src/` directory in the following order:

1. **Pre-Process the Data:** Handles missing values, removes outliers, and generates the processed dataset.
    ```bash
    cd src
    python preprocess.py
    ```

2. **Explore Customer Behavior:** Generates demographic insights and spending pattern visualizations.
    ```bash
    python customer_behavior.py
    ```

3. **Segmentation & Clustering:** Applies K-Means and Agglomerative Clustering to group customers and visualizes the PCA-reduced dimensions.
    ```bash
    python clustering.py
    ```

## 🧠 Methodology

1. **Data Preprocessing:** 
   - Handled categorical and numerical missing values.
   - Identified and removed standard outliers utilizing IQR (Interquartile Range).
2. **Exploratory Data Analysis (EDA):** 
   - Investigated age vs. purchase amount and total spending habits across different categories.
3. **Clustering Techniques:** 
   - Feature Scaling with `StandardScaler`.
   - **K-Means Clustering** and **Agglomerative Hierarchical Clustering**.
   - Used **PCA (Principal Component Analysis)** to reduce data into two principal components for clear 2D visualization of the customer segments.

## 📄 Documentation

For an in-depth explanation of the business context, methodology, findings, and managerial recommendations, please refer to:
- [Project Report (PDF)](./reports/Project_1_Report.pdf)
- [Presentation (PPTX)](./reports/Project_1_Presentation.pptx)

---
<p align="center">
  <i>Developed with ❤️ for Data Science.</i>
</p>
