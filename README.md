# Anemia Level Prediction in Children: Machine Learning Classification Analysis

## Project Overview

This project applies supervised machine learning techniques to predict anemia severity levels in children using demographic, socioeconomic, and clinical features. The study demonstrates the application of computational learning methods in healthcare, specifically for early anemia detection and risk assessment.

## Objective

Develop and compare classification models to predict anemia levels in pediatric populations, enabling healthcare professionals to identify at-risk children based on readily available demographic and clinical indicators.

## Dataset & Preprocessing

### Data Characteristics
- **Total Records**: 9,856 observations
- **Training Set**: 7,392 samples (75%)
- **Test Set**: 2,464 samples (25%)
- **Target Variable**: Anemia levels (4 classes: Not anemic, Mild, Moderate, Severe)

### Feature Engineering
- **Age Groups**: Converted to interval means
- **Residence Type**: Binary encoding (Urban=1, Rural=0)
- **Education Level**: Mapped to years of education required
- **Wealth Index**: Ordinal encoding (0=Poorest, 4=Richest)
- **Clinical Variables**: Standardized Yes/No responses to binary
- **Class Balancing**: Applied RandomOverSampler for improved distribution

### Data Cleaning
- Removed records with missing values using `drop.na()`
- Eliminated irrelevant features (marital status, breastfeeding timing)
- Converted all features to numerical format (int64/float64)

## Machine Learning Models

### 1. Support Vector Classification (SVC)
- **Algorithm**: Finds optimal hyperplane maximizing class margin
- **Kernel**: Handles non-linear data mapping
- **Performance**: 92% precision, recall, and F1-score
- **Error Analysis**: 61 false positives, 9 false negatives

### 2. K-Nearest Neighbors (KNN)  
- **Algorithm**: Classifies based on similarity to K nearest neighbors
- **Approach**: Groups similar data points for classification
- **Performance**: 92% precision, recall, and F1-score
- **Error Analysis**: 71 false positives, 38 false negatives

### 3. Random Forest Classifier
- **Algorithm**: Ensemble method with multiple decision trees
- **Optimization**: RandomizedSearchCV for hyperparameter tuning
- **Best Parameters**: n_estimators=100
- **Performance**: 95% precision, recall, and F1-score
- **Error Analysis**: 58 false positives, 2 false negatives

## Results Summary

| Model | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| SVC | 0.92 | 0.92 | 0.92 | 2,464 |
| KNN | 0.92 | 0.92 | 0.92 | 2,464 |
| **Random Forest** | **0.95** | **0.95** | **0.95** | **2,464** |

## Key Features

- **Correlation Analysis**: Identified minimal linear dependence between predictors
- **Multi-class Classification**: 4-level anemia severity prediction
- **Model Comparison**: Comprehensive evaluation of three ML algorithms  
- **Hyperparameter Optimization**: Grid search for optimal model configuration
- **Clinical Validation**: Confusion matrices for detailed error analysis
- **Class Distribution Analysis**: Histogram and probability plots

## Clinical Applications

- Early anemia screening in pediatric populations
- Risk stratification based on demographic factors
- Healthcare resource allocation optimization  
- Preventive care program targeting
- Public health surveillance support

## Technical Implementation

- **Language**: Python
- **Key Libraries**: scikit-learn, pandas, numpy
- **Evaluation Metrics**: Precision, Recall, F1-score, Confusion Matrix
- **Validation**: Train-test split with comprehensive model diagnostics

## Model Performance Insights

The Random Forest Classifier achieved the highest performance (95% accuracy) due to its ensemble approach and ability to handle complex feature interactions. While this creates a "black box" model with limited interpretability, the superior predictive capability makes it valuable for clinical decision support systems.
