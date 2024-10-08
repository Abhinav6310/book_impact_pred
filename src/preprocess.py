import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import joblib
from scipy.sparse import hstack, csr_matrix

class BookDataProcessor:
    def __init__(self, filepath):
        """
        Initialize the BookDataProcessor class.

        Args:
            filepath (str): Path to the CSV file containing book data.
        """
        self.df = pd.read_csv(filepath)
        self.df.drop(columns=["Unnamed: 0"], inplace=True)
        self.encoder = None

    @staticmethod
    def parse_date(date_str):
        """
        Parse the date string to a datetime object.

        Args:
            date_str (str): Date string to be parsed.

        Returns:
            pd.Timestamp or pd.NaT: Parsed datetime object or NaT if parsing fails.
        """
        if type(date_str) == str:
            date_str = re.sub(r'[^0-9T\-:Z+.]', '', date_str)
        try:
            parsed_date = pd.to_datetime(date_str, errors='coerce', infer_datetime_format=True)
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(date_str, format='%Y', errors='coerce')
                if parsed_date is not None:
                    parsed_date = pd.Timestamp(year=parsed_date.year, month=1, day=1)
            return parsed_date
        except:
            return pd.NaT

    def clean_data(self):
        """
        Clean and preprocess the book data by parsing dates, filling missing values, and preparing categorical features.
        """
        self.df["publishedDateCleaned"] = self.df["publishedDate"].swifter.apply(self.parse_date)
        self.df['publishedDateCleaned'] = pd.to_datetime(self.df['publishedDateCleaned'], utc=True)
        self.df['Year'] = self.df['publishedDateCleaned'].dt.year
        self.df["categories"] = self.df["categories"].apply(lambda x: eval(x))
        self.df["authors_cleaned"] = self.df["authors"].apply(lambda x: eval(x) if type(x) == str else np.nan)
        self.df['Year'] = self.df['Year'].fillna(int(self.df["Year"].mean()))

    def get_top_authors(self, n=100):
        """
        Get the top n authors based on their frequency in the dataset.

        Args:
            n (int, optional): Number of top authors to retrieve. Default is 100.

        Returns:
            list: A list of top n authors.
        """
        all_authors = [item for sublist in self.df['authors_cleaned'].dropna() for item in sublist]
        author_counts = Counter(all_authors)
        return [item for item, count in author_counts.most_common(n)]

    @staticmethod
    def one_hot_encode_list(items, top_100_authors):
        """
        One-hot encode a list of items based on the top 100 authors.

        Args:
            items (list or float): List of author names or NaN.
            top_100_authors (list): List of top 100 author names.

        Returns:
            dict: A dictionary representing one-hot encoded authors.
        """
        encoded = {item: 0 for item in top_100_authors}
        encoded['Other'] = 0

        if isinstance(items, float) and pd.isna(items):
            encoded['Other'] = 1
        else:
            for item in items:
                if item in top_100_authors:
                    encoded[item] = 1
                else:
                    encoded['Other'] = 1

        return encoded

    def encode_authors(self, top_100_authors):
        """
        Encode the authors using one-hot encoding based on the top 100 authors.

        Args:
            top_100_authors (list): List of top 100 author names.

        Returns:
            csr_matrix: Sparse matrix of one-hot encoded authors.
        """
        encoded_authors = self.df['authors_cleaned'].swifter.apply(
            lambda x: self.one_hot_encode_list(x, top_100_authors))
        encoded_df = pd.DataFrame(list(encoded_authors))
        return csr_matrix(encoded_df.values)

    def encode_publishers_and_categories(self):
        """
        Encode the publishers and categories using one-hot encoding.

        Returns:
            csr_matrix: Sparse matrix of one-hot encoded publishers and categories.
        """
        top_pub = self.df['publisher'].value_counts().nlargest(200).index
        self.df['top_publisher'] = self.df['publisher'].apply(lambda x: x if x in top_pub else 'Others')
        self.df["categories"] = self.df["categories"].apply(lambda x: x[0])

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(self.df[['top_publisher', 'categories']])
        joblib.dump(self.encoder, 'encoder.joblib')  # Save encoder for future use

        encoded_data_pub = self.encoder.transform(self.df[['top_publisher', 'categories']])
        return encoded_data_pub

    def get_features(self):
        """
        Get the features for training, including categorical features, year feature, and label.

        Returns:
            tuple: A tuple containing categorical features, year feature, label, and the dataframe.
        """
        top_100_authors = self.get_top_authors()
        sparse_authors = self.encode_authors(top_100_authors)
        encoded_pub_cat = self.encode_publishers_and_categories()

        cat_feat = hstack([encoded_pub_cat, sparse_authors])
        year_feature = self.df[['Year']].values
        label = list(self.df["Impact"])
        return cat_feat, year_feature, label, self.df