import torch.nn as nn
import torch

class BertRegressionModel(nn.Module):
    def __init__(self, bert_model, cat_feature_size, hidden_size=768):
        """
        Initialize the BertRegressionModel.

        Args:
            bert_model (transformers.BertModel): Pre-trained BERT model.
            cat_feature_size (int): Size of the categorical features.
            hidden_size (int, optional): Size of the hidden layer from BERT. Default is 768.
        """
        super(BertRegressionModel, self).__init__()
        self.bert = bert_model
        self.cat_feature_size = cat_feature_size
        self.fc_cat = nn.Linear(cat_feature_size, 16)
        self.fc_year = nn.Linear(1, 2)
        self.fc_combined = nn.Linear(hidden_size + 16 + 2, 1)

    def forward(self, input_ids, attention_mask, cat_feature, year):
        """
        Forward pass for the BertRegressionModel.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks for input tokens.
            cat_feature (torch.Tensor): Tensor of categorical features.
            year (torch.Tensor): Tensor containing the year feature.

        Returns:
            torch.Tensor: Output tensor containing the regression prediction.
        """
        # BERT output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # [CLS] token embedding (shape: [batch_size, hidden_size])

        # Process categorical features
        cat_output = torch.relu(self.fc_cat(cat_feature))  # (shape: [batch_size, 1, 16])
        cat_output = cat_output.squeeze(1)  # Remove the extra dimension to make it [batch_size, 16]

        # Process year feature
        year_output = torch.relu(self.fc_year(year.unsqueeze(1)))  # (shape: [batch_size, 2])

        # Concatenate BERT output, categorical features, and year feature
        combined_input = torch.cat((pooled_output, cat_output, year_output), dim=1)

        # Final regression output
        output = self.fc_combined(combined_input)
        return output