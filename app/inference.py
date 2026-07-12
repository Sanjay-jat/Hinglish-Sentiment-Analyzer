import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import json
import re
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

with open(os.path.join(MODEL_DIR, 'config.json')) as f:
    config = json.load(f)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(MODEL_DIR, 'hinglish_sp.model'))


class HinglishCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=3,
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        pooled_results = []
        for conv in self.convs:
            conv_out = conv(embedded)
            activated = F.relu(conv_out)
            pooled = F.max_pool1d(activated, kernel_size=activated.shape[2])
            pooled = pooled.squeeze(2)
            pooled_results.append(pooled)
        concatenated = torch.cat(pooled_results, dim=1)
        dropped = self.dropout(concatenated)
        logits = self.fc(dropped)
        return logits
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HinglishCNN(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    num_classes=config['num_classes'],
    num_filters=config['num_filters'],
    filter_sizes=config['filter_sizes'],
    dropout=config['dropout'],
    pad_id=config['pad_id']
)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'hinglish_cnn.pt'), map_location=device))
model.to(device)
model.eval()

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def encode_and_pad(text, max_len=config['max_len'], pad_id=config['pad_id']):
    ids = sp.encode(text, out_type=int)
    ids = ids[:max_len]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids

def predict(text):
    cleaned = clean_text(text)
    encoded = encode_and_pad(cleaned)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    with torch.no_grad():
        logits=model(input_tensor)
        probs=F.softmax(logits,dim=1)
        confidence,pred_idx=torch.max(probs,dim=1)
    label = config['label_map'][str(pred_idx.item())]
    return {
        'label': label,
        'confidence': round(confidence.item(), 4)
    }

if __name__ == '__main__':
    print(predict("यह movie बहुत अच्छी है"))
    print(predict("worst movie है ये"))
    print(predict("आज weather ठीक not है"))