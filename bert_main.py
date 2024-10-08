import os
from src.preprocess import BookDataProcessor
from src.bert_data_prep import DatasetPreparation
from src.train_bert import ModelTrainingBert

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def execute_bert_pipeline(filepath):
    num_epochs = 5
    batch_size = 16
    val_steps = 500

    processor = BookDataProcessor(filepath)
    processor.clean_data()
    cat_feat, year_feature, label, df = processor.get_features()

    dataset_preparation = DatasetPreparation(df, cat_feat, year_feature, label='Impact')
    train_dataset, val_dataset, test_dataset = dataset_preparation.create_datasets()

    model_trainer = ModelTrainingBert(train_dataset, val_dataset, test_dataset, model_name='bert-base-uncased')
    model_trainer.train_model(epochs = num_epochs, batch_size = batch_size, val_steps = val_steps)

if __name__ == "__main__":
    execute_bert_pipeline("books_task.csv")