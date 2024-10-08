import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class ModelTraining:
    def __init__(self, combined_features, label):
        """
        Initialize the ModelTraining class.

        Args:
            combined_features (csr_matrix): Combined features for training the model.
            label (array-like): Target labels for training the model.
        """
        self.combined_features = combined_features
        self.label = label
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()

    def split_data(self):
        """
        Split the data into training, validation, and test sets.

        Returns:
            tuple: A tuple containing training, validation, and test sets for features and labels.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.label, test_size=0.2, random_state=np.random.seed(0))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=np.random.seed(0))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def mean_absolute_percentage_error(self, y_true, y_pred, epsilon=1e-10):
        """
        Calculate the Mean Absolute Percentage Error (MAPE).

        Args:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.
            epsilon (float, optional): Small value to avoid division by zero. Default is 1e-10.

        Returns:
            float: The MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    def evaluate_model(self, y_train, y_train_pred, y_test, y_test_pred, model_name):
        """
        Evaluate the model's performance on the training and test datasets.

        Args:
            y_train (array-like): True values for the training set.
            y_train_pred (array-like): Predicted values for the training set.
            y_test (array-like): True values for the test set.
            y_test_pred (array-like): Predicted values for the test set.
            model_name (str): Name of the model being evaluated.
        """
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_mape = self.mean_absolute_percentage_error(y_train, y_train_pred)
        test_mape = self.mean_absolute_percentage_error(y_test, y_test_pred)

        print(f"\nModel: {model_name}")
        print(f"Val RMSE: {train_rmse:.4f}, Val MAPE: {train_mape:.2f}%")
        print(f"Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%")

    def train_and_evaluate_models(self):
        """
        Train and evaluate regression models using the training, validation, and test sets.
        """
        # Train Ridge Regression model
        ridge_model = Ridge(alpha=1)
        ridge_model.fit(self.X_train, self.y_train)

        # Predict on validation and test sets
        y_val_pred_ridge = ridge_model.predict(self.X_val)
        y_test_pred_ridge = ridge_model.predict(self.X_test)

        # Evaluate Ridge Regression model
        self.evaluate_model(self.y_val, y_val_pred_ridge, self.y_test, y_test_pred_ridge, "Ridge Regression")

        # Save the trained model
        joblib.dump(ridge_model, 'ridge_model.joblib')