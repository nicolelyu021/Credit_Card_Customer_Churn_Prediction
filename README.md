# Credit Card Customer Churn Prediction

## Project Overview 🎯
Built machine learning models to identify credit card customers likely to churn, achieving 97.53% accuracy using XGBoost. 
- Analyzed dataset of 10,127 customer records
- Implemented and compared Logistic Regression, Random Forest and XGBoost models
- Handled class imbalance through oversampling and class weights
- Feature importance analysis for business insights

## Key Results
- Best Model (XGBoost): 97.53% accuracy, 94.71% precision, 90.96% recall
- Key churn indicators identified: transaction patterns, credit utilization, and customer activity levels
- Successfully handled class imbalance (8500 existing vs 1627 churned customers)

## Technical Implementation

### 1. Data Exploration & Preprocessing ([explore_data.ipynb](notebooks/1_explore_data.ipynb))
- Initial analysis of 10,127 records with 23 features
- Feature correlation analysis
- Distribution analysis and class imbalance identification
- Data quality checks and cleaning

### 2. Feature Engineering & Visualization ([feature_engineering.ipynb](notebooks/2_feature_engineering.ipynb))
- Comprehensive data visualization:
  - Correlation heatmaps for feature relationships
  - Distribution plots for key variables
  - Box plots for outlier detection
  - Scatter plots for relationship analysis
- Feature engineering and preprocessing:
  - Handled severe class imbalance through:
    - Random oversampling (balanced)
    - Class weights (1:3.167 ratio)
    - SMOTE implementation
  - Data splitting: 70% training, 20% test, 10% validation
  - Feature scaling and encoding

### 3. Model Development ([models.ipynb](notebooks/3_models.ipynb))
Implemented and compared three models:
1. **Logistic Regression**: Baseline model
   - Base: 88.25% accuracy, 74.58% precision
   - With Oversampling: 82.63% accuracy, 81.92% recall
   
2. **Random Forest**: Strong performer
   - Base: 92.10% accuracy, 95.33% precision
   - With Oversampling: 96.05% accuracy, 89.14% precision
   - Final Tuned: 96.84% accuracy, F1 Score: 0.90

3. **XGBoost**: Best performing model
   - Base: 96.54% accuracy, 92.77% precision
   - Final Tuned: 97.53% accuracy, 94.71% precision, 90.96% recall
   - Best Parameters: 
     - Learning rate: 0.1
     - Max depth: 3
     - N_estimators: 300

## Top 3 Most Important Features Finding
1. Total Transaction Count
2. Total Transaction Amount
3. Total Change in Transaction Count (Q4 vs Q1)

## Notebooks Structure
1. **Data Exploration** ([explore_data.ipynb](notebooks/1_explore_data.ipynb))
   - Initial data analysis
   - Class imbalance investigation
   - Feature distributions and correlations
   
2. **Feature Engineering** ([feature_engineering.ipynb](notebooks/2_feature_engineering.ipynb))
   - Data visualization and insights
   - Class imbalance handling
   - Data preprocessing
   - Train/test/validation split
   
3. **Model Development** ([models.ipynb](notebooks/3_models.ipynb))
   - Model implementation
   - Hyperparameter tuning
   - Performance comparison
   - Feature importance analysis

## Tools & Technologies
- Python
- Libraries: pandas, numpy, scikit-learn, XGBoost
- Feature Engineering & Model Selection
- Hyperparameter Tuning
- Model Evaluation Metrics

## Business Impact
- Early identification of potential churners with 97.53% accuracy
- High precision (94.71%) reduces false positives in customer retention efforts
- Strong recall (90.96%) ensures most at-risk customers are identified
- Clear insights into key churn indicators for business strategy
