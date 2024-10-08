from src.preprocess import BookDataProcessor
from src.regression_data_prep import CombinedFeatureProcessor
from src.train_regression import ModelTraining

def execute_pipeline(filepath):
    processor = BookDataProcessor(filepath)
    processor.clean_data()
    cat_feat, year_feature, label, df = processor.get_features()

    feature_processor = CombinedFeatureProcessor(df, cat_feat, year_feature)
    feature_processor.clean_text_data()
    combined_features = feature_processor.get_combined_features()

    model_trainer = ModelTraining(combined_features, label)
    model_trainer.train_and_evaluate_models()

if __name__ == "__main__":
    execute_pipeline("books_task.csv")