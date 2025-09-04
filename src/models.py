import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

class EntityAttentionPooling(nn.Module):
    """
    Pulling via attention on entity tokens
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, entity_hidden_states, entity_mask):
        scores = self.attention(entity_hidden_states).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~entity_mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = (entity_hidden_states * weights).sum(dim=1)  # [B, H]
        return pooled


class BertEntitySentimentClassifier(nn.Module):
    """
    Entity sentiment classifier with attention pooling
    """
    def __init__(self, model_checkpoint, tokenizer, num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.pooling = EntityAttentionPooling(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, entity_mask, labels=None):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.pooling(hidden_states, entity_mask)
        logits = self.classifier(pooled)
    
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
    
        return SequenceClassifierOutput(loss=loss, logits=logits)