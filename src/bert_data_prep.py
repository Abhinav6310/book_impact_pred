import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, texts, years, cat_features, labels=None):
        """
        Initialize the CustomDataset.

        Args:
            texts (array-like): List or array of text data.
            years (array-like): List or array of year data.
            cat_features (array-like): Categorical features in dense format.
            labels (array-like, optional): List or array of labels. Default is None.
        """
        self.texts = texts
        self.years = years
        self.cat_features = cat_features
        self.labels = labels

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the text, year, categorical features, and optionally the label.
        """
        item = {}
        item['text'] = self.texts[idx]
        item['year'] = torch.tensor(self.years[idx], dtype=torch.float32)
        item['cat_feature'] = torch.tensor(self.cat_features[idx].todense(), dtype=torch.float32)

        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class DatasetPreparation:
    def __init__(self, df, cat_feat, year_feature, label):
        """
        Initialize the DatasetPreparation class.

        Args:
            df (DataFrame): The input dataframe containing book data.
            cat_feat (array-like): Categorical features.
            year_feature (array-like): Year feature.
            label (str): Column name of the target label.
        """
        self.df = df
        self.cat_feat = cat_feat
        self.year_feature = year_feature
        self.label = label

    def prepare_text_data(self):
        """
        Prepare the text data and fill missing values.

        Returns:
            tuple: A tuple containing texts, years, and labels.
        """
        self.df['text'] = self.df['Title'] + " " + self.df['description'].fillna("")
        self.df['Year'] = self.df['Year'].fillna(int(self.df["Year"].mean()))
        texts = self.df['text'].values
        years = self.df['Year'].values
        labels = list(self.df[self.label])
        return texts, years, labels

    def split_data(self, texts, years, labels):
        """
        Split the data into training, validation, and test sets.

        Args:
            texts (array-like): List or array of text data.
            years (array-like): List or array of year data.
            labels (array-like): List or array of labels.

        Returns:
            tuple: A tuple containing training, validation, and test sets for texts, years, labels, and categorical features.
        """
        X_train, X_test, y_train, y_test, years_train, years_test, cat_train, cat_test = train_test_split(
            texts, labels, years, self.cat_feat, test_size=0.2, random_state=np.random.seed(0)
        )
        X_train, X_val, y_train, y_val, years_train, years_val, cat_train, cat_val = train_test_split(
            X_train, y_train, years_train, cat_train, test_size=0.15, random_state=np.random.seed(0)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, years_train, years_val, years_test, cat_train, cat_val, cat_test

    def create_datasets(self):
        """
        Create PyTorch datasets for training, validation, and testing.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        """
        texts, years, labels = self.prepare_text_data()
        X_train, X_val, X_test, y_train, y_val, y_test, years_train, years_val, years_test, cat_train, cat_val, cat_test = self.split_data(
            texts, years, labels)

        train_dataset = CustomDataset(X_train, years_train, cat_train, y_train)
        val_dataset = CustomDataset(X_val, years_val, cat_val, y_val)
        test_dataset = CustomDataset(X_test, years_test, cat_test, y_test)

        return train_dataset, val_dataset, test_dataset