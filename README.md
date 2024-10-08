### This assignment involved building regression models using textual and categorical features. We explore two approaches: a Ridge Regression model with TF-IDF vectorization for text, and a custom BERT-based architecture, integrating text embeddings with categorical data for prediction.
 
#### 1. **Title**:
- **Type**: String feature
- **Missing Values**: None
- **Handling**:
  - The `Title` feature is a string with no missing values. It will be combined with the `Description` feature as part of text preprocessing.

---

#### 2. **Description**:
- **Type**: String feature
- **Missing Values**: Some missing values
- **Handling**:
  - Since we are concatenating the `Title` and `Description`, missing values are not explicitly handled. A `<sep>` token is used during concatenation to indicate missing descriptions.
  - **Alternative**: Missing descriptions could be filled with a default value like "No description" for transformer based model as it has language understanding.

---

#### 3. **Published Date**:
- **Type**: Date feature
- **Missing Values**: Some missing values and inconsistent formats
- **Handling**:
  - We extracted the **year** from the date feature, as it was the most consistently available component. For missing years, the mean year was imputed and passed to the model.
  - **Alternative**: Additional components like month and day could also be extracted and used. Cyclical encoding (e.g., sine/cosine) could be applied to capture seasonality.

---

#### 4. **Categories**:
- **Type**: Categorical feature (List of categories)
- **Missing Values**: None
- **Handling**:
  - We applied **one-hot encoding** for around 100 unique values in this feature.
  - **Alternative**: Frequency encoding or target encoding (encoding based on the relationship to the target variable) could provide more informative representations, especially for categories with sparse occurrences.

---

#### 5. **Authors**:
- **Type**: Categorical feature (List of authors)
- **Missing Values**: Some (due to infrequent appearances)
- **Handling**:
  - Direct one-hot encoding was avoided due to the high number of unique authors (97,801). Instead, we encoded the **top N authors** and grouped the rest into an "Other" category to reduce dimensionality.
  - **Alternative**: Authors could be encoded based on their frequency of occurrence or the average impact (target) they contribute to, providing a more informative representation.

---

#### 6. **Publisher**:
- **Type**: Categorical feature
- **Missing Values**: None
- **Handling**:
  - Similar to `Authors`, we encoded the **top N publishers** and grouped the rest into "Other."
  - **Alternative**: Frequency or target encoding based on the relationship to the impact score could also be considered.

---

#### 7. **Impact**:
- **Type**: Numeric (Target Variable)
- **Missing Values**: None
- **Statistics**:
  - **Standard Deviation**: 63.64
  - **Mean**: 786.76
  - The target feature (`Impact`) has no missing values, and no additional preprocessing was needed.

---

### Model 1: Ridge Regression (Base Model)

- **Model Type**: Linear Regression with L2 regularization (Ridge)
- **Text Preprocessing**:
  - Textual features (`Title` + `Description`) were combined and vectorized using **TF-IDF** with a limit of 5000 features due to hardware constraints.
  - Preprocessing steps included **stopword removal** and **lemmatization** to improve the quality of the vectorized text.
  
- **Other Features**:
  - Non-text features (`Year`, `Categories`, `Authors`, `Publisher`) were used as described in the handling section above.

- **Model Performance**:
  - Despite the preprocessing and feature engineering, the Ridge Regression model did not perform well on the validation and test sets.
  - **Results**:
    - **Validation RMSE**: 61.7134
    - **Validation MAPE**: 5.74%
    - **Test RMSE**: 61.4327
    - **Test MAPE**: 5.72%

- **Analysis**:
  - Ridge Regression struggled to capture the complexity of the high-dimensional text features and other categorical features.
  - **Alternative**: More advanced models, such as **XGBoost** or other tree-based models, could better capture feature interactions, especially with high-dimensional sparse text features.

---

### Transition to Transformer-Based Model (BERT)

Due to the limitations observed in Ridge Regression, we transitioned to a transformer-based model to incorporate the text features more effectively.

#### Model 2: BERT Regression Model with Additional Features

- **Architecture Overview**:
  - We used a **custom architecture** that integrates **BERT embeddings** with categorical and numerical features.
  - **BERT** processed the textual data, generating embeddings that were concatenated with the one-hot encoded categorical features and the `Year` feature.
  - These combined features were passed through fully connected layers to produce the final regression output.
  
- **Training**:
  - The model was trained using **Mean Squared Error (MSE) loss**, which is appropriate for regression tasks, ensuring that the model learns from both the textual and structured features.

- **Text Length Handling**:
  - **Max Token Length**: The BERT model has a maximum token length of 512, and some descriptions exceeded this limit. To handle this, we truncated the text beyond the 512-token limit.
  - **Alternative**: In the future, we could use text summarization techniques (e.g., **T5**, **GPT**, **LLaMA**) to capture the full context of long descriptions without truncation.

- **Model Performance**:
  - **Validation RMSE**: 63.9729
  - **Validation MAPE**: 6.02%
  - **Test RMSE**: 63.4974
  - **Test MAPE**: 5.96%

- **Conclusion**:
  - Even with a more complex architecture that leverages BERT for text features, the model performance did not improve significantly over the base Ridge Regression model.
  - This indicates that the `Impact` target variable may not have a strong relationship with the provided features, potentially implying randomness in the target or that important features are missing from the dataset.

