import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm.notebook import tqdm
from bert_model_arch import BertRegressionModel

class ModelTrainingBert:
    def __init__(self, train_dataset, val_dataset, test_dataset, model_name):
        """
        Initialize the ModelTrainingBert class.

        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The test dataset.
            model_name (str): The name of the pre-trained BERT model to be used.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.bert_model = BertModel.from_pretrained(model_name)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        cat_feature_size = train_dataset.cat_features.shape[1]
        self.model = BertRegressionModel(bert_model=self.bert_model, cat_feature_size=cat_feature_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.criterion = nn.MSELoss()
        self.patience = 3
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

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

    def evaluate_model(self, dataloader):
        """
        Evaluate the model on the provided dataset.

        Args:
            dataloader (DataLoader): DataLoader for the dataset to be evaluated.

        Returns:
            tuple: A tuple containing RMSE and MAPE values.
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt",
                                        max_length=512).to(self.device)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                cat_feature = batch['cat_feature'].to(self.device)
                year = batch['year'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cat_feature=cat_feature,
                                     year=year)

                predictions.extend(outputs.squeeze().tolist())
                true_labels.extend(labels.tolist())

        mse = mean_squared_error(true_labels, predictions)
        rmse = np.sqrt(mse)
        mape = self.mean_absolute_percentage_error(true_labels, predictions)
        print(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        return rmse, mape

    def train_model(self, epochs, batch_size, val_steps):
        """
        Train the model using the provided training and validation datasets.

        Args:
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            val_steps (int): Number of steps after which to evaluate the model on the validation set.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size * 2, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            step = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                step += 1
                inputs = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt",
                                        max_length=512).to(self.device)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                cat_feature = batch['cat_feature'].to(self.device)
                year = batch['year'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cat_feature=cat_feature,
                                     year=year)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"Train Loss": f"{total_loss / step:.4f}"})
                if step % val_steps == 0:
                    val_rmse, val_mape = self.evaluate_model(val_loader)
                    print(f"Step {step}, Validation RMSE: {val_rmse:.4f}, Validation MAPE: {val_mape:.2f}%")
                    self.model.train()

            val_rmse, val_mape = self.evaluate_model(val_loader)
            print(f"Epoch {epoch + 1}, Validation RMSE: {val_rmse:.4f}, Validation MAPE: {val_mape:.2f}%")

            if val_rmse < self.best_val_loss:
                self.best_val_loss = val_rmse
                self.epochs_without_improvement = 0
                print(f"Validation loss improved to {val_rmse:.4f}, saving model...")
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement in validation loss for {self.epochs_without_improvement} epoch(s).")

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs due to no improvement.")
                break

        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        test_rmse, test_mape = self.evaluate_model(test_loader)
        print(f"Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%")