import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class CombinedFeatureProcessor:
    def __init__(self, df, cat_feat, year_feat):
        """
        Initialize the CombinedFeatureProcessor class.

        Args:
            df (DataFrame): The input dataframe containing book data.
            cat_feat (csr_matrix): Categorical features in sparse matrix format.
            year_feat (array-like): Year feature.
        """
        self.df = df
        self.cat_feat = cat_feat
        self.year_feat = year_feat

    @staticmethod
    def clean_and_lemmatize(text):
        """
        Clean and lemmatize the input text.

        Args:
            text (str): The text to be cleaned and lemmatized.

        Returns:
            str: The cleaned and lemmatized text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        cleaned_text = ' '.join(words)
        return cleaned_text

    def clean_text_data(self):
        """
        Clean and preprocess the text data in the dataframe by combining title and description.
        """
        self.df['text'] = self.df['Title'] + " " + self.df['description'].fillna("")
        self.df['cleaned_text'] = self.df['text'].swifter.apply(self.clean_and_lemmatize)

    def get_tfidf_features(self):
        """
        Generate TF-IDF features from the cleaned text data.

        Returns:
            csr_matrix: Sparse matrix of TF-IDF features.
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_features = tfidf_vectorizer.fit_transform(self.df['cleaned_text'])
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')  # Save vectorizer for future use
        return tfidf_features

    def get_combined_features(self):
        """
        Combine the TF-IDF features with categorical and year features.

        Returns:
            csr_matrix: Sparse matrix of combined features.
        """
        tfidf_features = self.get_tfidf_features()
        combined_features = hstack([tfidf_features, self.cat_feat, self.year_feat])
        return combined_features