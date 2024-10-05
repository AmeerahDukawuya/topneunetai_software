# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.weightstats import ztest
import plotly.express as px

# Set the title of the app
st.title("Advanced Statistical Analysis App")

# Sidebar for file upload and navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ("Home", "Descriptive Statistics", "Inferential Statistics", "Statistical Tests", "Visualization")
)

# Home Page
if option == "Home":
    st.subheader("Welcome to the Advanced Statistical Analysis App")
    st.write("""
        This app allows you to perform various types of statistical analysis:
        - Descriptive Statistics
        - Inferential Statistics
        - Statistical Tests
        - Data Visualization
    """)
    st.write("To get started, upload your dataset and navigate to the desired analysis section.")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset uploaded successfully!")

    # Extract numeric and categorical columns for use throughout the app
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Descriptive Statistics Section
if option == "Descriptive Statistics":
    st.subheader("Descriptive Statistics")
    
    if uploaded_file is not None:
        st.write("**Preview of the dataset:**")
        st.dataframe(df.head())

        # Allow users to select columns for descriptive statistics
        selected_columns = st.multiselect("Select numeric columns for analysis", numeric_columns)

        if selected_columns:
            st.write("**Descriptive Statistics:**")
            st.write(df[selected_columns].describe())

            # Visualize distributions with histograms
            st.subheader("Histograms of Selected Columns")
            for col in selected_columns:
                st.write(f"Distribution of {col}")
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
        else:
            st.write("Please select at least one numeric column.")
    else:
        st.write("Please upload a dataset to use this feature.")

# Inferential Statistics Section
if option == "Inferential Statistics":
    st.subheader("Inferential Statistics")

    if uploaded_file is not None:
        st.write("This section helps with hypothesis testing and estimation.")
        
        # Select two numeric columns for a T-test or Z-test
        if len(numeric_columns) > 1:
            col1, col2 = st.selectbox("Select two columns for hypothesis testing", numeric_columns, index=[0, 1], key="test_columns")

            if st.button("Perform T-test"):
                t_stat, p_value = stats.ttest_ind(df[col1], df[col2], nan_policy='omit')
                st.write(f"T-test results between {col1} and {col2}:")
                st.write(f"T-statistic: {t_stat}, P-value: {p_value}")

            if st.button("Perform Z-test"):
                z_stat, p_val = ztest(df[col1].dropna(), df[col2].dropna())
                st.write(f"Z-test results between {col1} and {col2}:")
                st.write(f"Z-statistic: {z_stat}, P-value: {p_val}")
        else:
            st.write("Please upload a dataset with at least two numeric columns.")

# Statistical Tests Section
if option == "Statistical Tests":
    st.subheader("Statistical Tests")

    if uploaded_file is not None:
        st.write("Select from common statistical tests:")
        st.write("You can choose to perform t-tests, chi-square tests, ANOVA, etc.")

        # Chi-square test
        if categorical_columns:
            col1, col2 = st.selectbox("Select two categorical columns for Chi-square test", categorical_columns, index=[0, 1], key="chi_square")

            if st.button("Perform Chi-square test"):
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2_stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-square Test between {col1} and {col2}:")
                st.write(f"Chi-square Statistic: {chi2_stat}, P-value: {p}")
        else:
            st.write("Please upload a dataset with categorical columns.")

# Visualization Section
if option == "Visualization":
    st.subheader("Data Visualizations")

    if uploaded_file is not None:
        st.write("Select columns for plotting visualizations:")
        selected_plot_columns = st.multiselect("Select numeric columns for visualization", numeric_columns)
        
        if len(selected_plot_columns) > 1:
            st.write("**Correlation Heatmap**")
            correlation = df[selected_plot_columns].corr()
            fig, ax = plt.subplots()
            sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # Pair plot visualization
            st.write("**Pair Plot**")
            fig = sns.pairplot(df[selected_plot_columns])
            st.pyplot(fig)

        else:
            st.write("Please select at least two columns to generate visualizations.")
    else:
        st.write("Please upload a dataset to use this feature.")
