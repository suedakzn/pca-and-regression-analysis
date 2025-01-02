# pca-and-regression-analysis

## Project Overview
This project uses **Principal Component Analysis (PCA)** for dimensionality reduction and builds a **Linear Regression model** to predict prices in the tourism sector. The goal is to simplify the dataset while retaining key information and demonstrate the effectiveness of PCA in reducing data complexity.
---

## Dataset
- **Source:** [Kaggle - Gezinomi Dataset](https://www.kaggle.com/datasets/merveoztiryaki/gezinomi/data)
- **Description:** 
  - This dataset contains tourism-related data, including:
    - `CheckInDate`: The check-in date of the booking.
    - `SaleDate`: The date of the sale.
    - `Price`: The price of the service.
    - Additional categorical variables.
- **Target Variable:** `Price`
---

## Objectives
1. Perform **data preprocessing** to prepare the dataset for analysis.
2. Use **PCA** to reduce the dataset's dimensions while retaining most of its variance.
3. Build a **Linear Regression model** using the PCA components.
4. Evaluate the model’s performance with metrics like **MAE**, **MSE**, and **R²**.
5. Visualize and interpret the results to extract insights.
6. Explore potential improvements and future steps for the project.

--- 
## Completed Steps
### 1. Data Preprocessing:
- Handled missing values by removing rows where `Price` was null.
- Applied **One-Hot Encoding** to transform categorical variables into numerical format.

### 2. PCA Analysis:
- Applied PCA to reduce the dimensionality of the dataset.
- Retained 4 components, which explained **100% of the variance**.
- The first 3 components explained **77.6% of the variance**, indicating that most of the information is captured in these components.

### 3. Linear Regression Model:
- Built a regression model using the PCA components as features.
- Evaluated the model's performance with the following results:
  - **Mean Absolute Error (MAE):** 3.31e-14
  - **Mean Squared Error (MSE):** 1.61e-27
  - **R² Score:** 1.0 (Perfect prediction)
- Visualized the model's predictions and residuals to verify its accuracy.

---

## Visualizations
The following visualizations were created to better understand the data and model performance:

1. **Actual vs Predicted Values:**
   - Visualizes the relationship between the model's predictions and the true values.

2. **Residual Distribution:**
   - Shows the distribution of errors, which are centered around zero, indicating minimal bias.

3. **Visualizing Model Performance as a Table**

4. **PCA Component Coefficients:**
   - Highlights the impact of each PCA component on the target variable.
5. **Cumulative Variance Explained by PCA Components:**
   - Demonstrates how much variance is retained as more components are added.

---

## Future Work
To further improve and expand this project, the following steps are planned:
1. **Model Generalization:**
   - Test the model on additional datasets or use cross-validation to evaluate its robustness.

2. **Experiment with Alternative Models:**
   - Implement and compare other models such as:
     - Random Forest Regressor
     - Gradient Boosting

3. **Feature Engineering:**
   - Explore creating new features from existing ones (e.g., time-based variables like `days_to_checkin`).

4. **Deployment:**
   - Convert the model into a web application or API using **Flask** or **Streamlit** for real-time predictions.

---
## Key Insights
The analysis revealed the following key insights:

1. **Importance of Dimensionality Reduction:**
   - PCA reduced the dataset's complexity while retaining 100% of its variance, making it an effective tool for high-dimensional datasets.

2. **Predictive Accuracy:**
   - The Linear Regression model achieved perfect predictions on the training data, demonstrating the strength of the PCA-transformed features.

3. **High Variance Explained by Fewer Components:**
   - The first three components explained 77.6% of the variance, highlighting the dataset's inherent structure.

These insights demonstrate the power of PCA in simplifying data and improving model performance.

---
## Contribution
This project was completed as an individual effort to enhance my data science skills. 
All steps, from data preprocessing to model building and evaluation, were implemented independently.

If you'd like to contribute or provide feedback, feel free to reach out!

---
## References
1. [Kaggle - Gezinomi Dataset](https://www.kaggle.com/datasets/merveoztiryaki/gezinomi/data)
2. Scikit-learn Documentation: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
3. Python Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
