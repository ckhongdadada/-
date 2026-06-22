"""
公共模型定义
============
主线实验脚本复用的模型类。
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self._feature_dim = 128 * 4 * 4

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1)


class BertClassifier(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: int,
        use_tfidf: bool = True,
        tfidf_dim: int = 2000,
        tfidf_hidden: int = 128,
    ):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(model_path)
        self.use_tfidf = use_tfidf

        bert_hidden = self.bert.config.hidden_size

        if use_tfidf:
            self.tfidf_fc = nn.Sequential(
                nn.Linear(tfidf_dim, tfidf_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            classifier_input = bert_hidden + tfidf_hidden
        else:
            classifier_input = bert_hidden

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input, num_classes),
        )

        self._feature_dim = bert_hidden

    def forward(self, input_ids, attention_mask, tfidf=None, return_features=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output

        if self.use_tfidf and tfidf is not None:
            tfidf_feat = self.tfidf_fc(tfidf)
            features = torch.cat([pooled, tfidf_feat], dim=-1)
        else:
            features = pooled

        logits = self.classifier(features)

        if return_features:
            return logits, pooled
        return logits

    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output


class TextMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._feature_dim = hidden_dim

    def forward(self, tfidf, return_features=False):
        features = self.feature_extractor(tfidf)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def get_features(self, tfidf):
        return self.feature_extractor(tfidf)


class TimeSeriesCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes),
        )
        self._feature_dim = hidden_dim * 4

    def forward(self, x, return_features=False):
        features = self.get_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        return self.features(x).flatten(1)


class LossPredictionModule(nn.Module):
    """Loss Prediction Module for Learning Loss Active Learning.

    Reference: Yoo & Kweon, "Learning Loss for Active Learning" (CVPR 2019).
    Predicts the loss value for each sample based on its feature representation.
    Samples with highest predicted loss are selected for annotation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict loss values from feature vectors.

        Args:
            features: (batch_size, feature_dim) feature tensor.
        Returns:
            (batch_size, 1) predicted loss values.
        """
        return self.fc(features)
