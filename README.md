![Car Price Prediction Project Logo](file-L7iarjRLxNakKbwDeDJCB2 (1).webp)

# 🚗 Car Price Prediction Project with Machine Learning 🤖

## ✨ Overview

This project leverages the power of machine learning to predict car prices based on various features. We've explored several algorithms, ultimately focusing on XGBoost for its exceptional performance. 🚀 This project encompasses data preprocessing, training, optimization, evaluation, and model deployment.

## 🗂️ Files Included

*   CarPrice_Assignment.csv: 📊 The dataset for training and evaluation.
*   car_pricing_prediction_model.pkl: 🧠 The trained machine learning model file, saved using joblib.
*   Car_Price_ML_Model.ipynb: ⚙️ The main Jupyter Notebook containing the training and evaluation logic.
*    API.py: 🌐 A simple Flask API for model predictions, along with Postman usage instructions.
*   README.md: 📖 This file that you're currently reading.
*   requirements.txt: 📚 A list of all required Python libraries for this project.

## 🛠️ Libraries Used

*   pandas: 🐼 For data manipulation and analysis using DataFrames.
*   matplotlib.pyplot: 📈 For creating static, interactive, and animated visualizations.
*   seaborn: 📊 For visually appealing statistical plots.
*   numpy: 🧮 For efficient numerical computation.
*   scikit-learn (sklearn): ⚙️ A comprehensive machine learning library that includes:
    *   Pipeline: 🔗 For building sequential data processing steps.
    *   SimpleImputer: 🧹 For handling missing values.
    *   StandardScaler: ⚖️ For scaling numerical features.
    *   OrdinalEncoder: 🏷️ For encoding categorical features.
    *   LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR: 🤖 Regression algorithms.
    *   mean_squared_error, r2_score: 📉 Performance evaluation metrics.
    *   cross_val_score: 🧪 For cross-validation.
    *   GridSearchCV: 🔍 For parameter tuning.
    *   StratifiedShuffleSplit: 🔀 For creating stratified train/test splits.
    *   ColumnTransformer: 🔀 For applying different transformations to different types of data.
*   joblib: 💾 For saving and loading machine learning models.
*   xgboost (xgb): 🌲 For the XGBoost gradient boosting algorithm.
*   scipy: 🔬 For advanced statistical analysis.
*   flask: 🌐 A lightweight web framework for the API.

## ⚙️ Project Workflow

1.  📥 Data Loading & Exploration:
    *   Loads the dataset from CarPrice_Assignment.csv using pandas.
    *   Displays the first few rows (head()), descriptive stats (describe()), and data information (info()).
    *   Prints unique values for categorical columns.
    *   Visualizes numerical data distributions using histograms (hist()).
2.  🔎 Outlier Detection:
    *   Creates boxen plots to visualize outliers for specified columns.
    *   Defines a function to detect outliers using the IQR method.
    *   Prints detected outlier values.
3.  📊 Correlation Analysis:
    *   Extracts numerical columns and calculates the correlation matrix.
    *   Sorts and displays correlation values with respect to price.
4.  📉 Relationship Visualization:
    *   Creates a scatter plot to visualize the relationship between enginesize and price.
5.  🗂️ Feature Binning:
    *   Bins numerical columns like enginesize, horsepower, and curbweight.
    *   Creates a combined bin column.
    *   Displays the distribution using histograms.
6.  🔀 Stratified Splitting:
    *   Splits the data into stratified train/test sets using StratifiedShuffleSplit.
    *   Uses the training set for subsequent steps.
7.  🧹 Data Preparation:
    *   Separates features (X) from target (y).
    *   Extracts numerical columns for transformation.
    *   Creates a numerical_pipeline with SimpleImputer and StandardScaler.
    *   Creates a ColumnTransformer to handle numerical and categorical columns separately using the previously created pipeline and OrdinalEncoder.
8. 🏋️ Model Training & Evaluation:
    * Creates regression models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR, xgb.XGBRegressor).
    * Trains each model with preprocessed data.
    * Evaluates each model's performance using mean_squared_error and cross-validation.
9.  ⚙️ Model Optimization:
    * Uses GridSearchCV to tune parameters for RandomForestRegressor and xgb.XGBRegressor.
    * Displays the best parameters and the best model found.
10. 🧪 Test Evaluation:
    * Extracts the best model from the grid search (XGBoost).
    * Prepares test data with the same preprocessing as the training data.
    * Uses the best model for predictions on the test set and calculates RMSE.
11. 📈 Confidence Interval:
     * Computes the confidence interval for RMSE using Student's t-distribution.
12. 🎯 Accuracy-like Percentage:
    * Computes an accuracy-like percentage to provide a more intuitive understanding of model performance.
13. 📦 Model Deployment:
     * Creates a complete pipeline that includes all preprocessing steps and the best performing model.
     * Saves the trained model to car_pricing_prediction_model.pkl using joblib.

## 🤖 Algorithms Used

*   Linear Regression: Simple model assuming a linear relationship. 📏
*   Decision Tree Regressor: Tree-like model capable of capturing complex relationships. 🌳
*   Random Forest Regressor: Ensemble of decision trees for robustness. 🌲🌲🌲
*   Support Vector Regression (SVR): Powerful algorithm effective in high-dimensional spaces. 💪
*   XGBoost Regressor: Gradient boosting algorithm that is very powerful and accurate. 🚀

🏆 Best Model: XGBoost outperformed other models after parameter tuning using GridSearchCV.

## 🚀 How to Use the Model

1.  Install Requirements:
   
    pip install -r requirements.txt
    
2.  Run the Project:
    *   To run the Jupyter Notebook: Open Car_Price_ML_Model.ipynb in Jupyter Notebook or Jupyter Lab.
    *   To launch the API:
       
        python API.py
        
3.  Using the Trained Model:
    *   Load car_pricing_prediction_model.pkl and use it for predictions in any project.
    *   Refer to Car_Price_ML_Model.ipynb for an example of how to load and use the model.

## 🔗 Pipeline Explanation

The Pipeline is a tool from scikit-learn that streamlines workflows, in this project:

*   numerical_pipeline:
    *   SimpleImputer(strategy='median'): Handles missing values.
    *   StandardScaler(): Scales numerical data.
*   main_pipeline:
    *   Applies the numerical_pipeline to numerical columns.
    *   Applies the OrdinalEncoder to categorical columns.
*   full_pipeline_with_predictor:
    *   Combines all data preprocessing steps and the best performing model.

## 🌐 API and Postman

*   The API.py file has a Flask API for model predictions.
*   Use Postman to interact with the API:
    *   URL: http://127.0.0.1:5000/predict
    *   Method: POST
    *   Body: JSON data resembling a row from CarPrice_Assignment.csv with your car features.

## 📝 Additional Notes

*   GridSearchCV was instrumental in model tuning for optimal performance.
*   Confidence intervals for RMSE are computed for reliable evaluation.
*   Accuracy-like percentage provides an interpretable metric for model evaluation.
*   All necessary libraries are listed in requirements.txt.

## 🎉 Conclusion

This project provides an end-to-end solution for predicting car prices using machine learning. With a carefully tuned XGBoost model, it offers a powerful tool for understanding the dynamics of the car market. Feel free to explore, contribute, and improve upon this project!

If you have any questions or need further assistance, don't hesitate to ask. 😉


## 👨🏻‍💻 Team Members

*   Fares Alhomazah: [LinkedIn Profile](https://www.linkedin.com/in/fares-abdulghani-alhomazah-6b1802288?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
*   Ahmed Aljaifi: [LinkedIn Profile](https://www.linkedin.com/in/ahmed-al-jaifi-ab213617a)
