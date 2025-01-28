import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def save_model_incrementally(model, base_name="word2vec_model"):
    # Verifica se o arquivo base existe, caso contrário adiciona um sufixo numérico
    i = 0
    while True:
        model_path = f"{base_name}{i}.pth" if i > 0 else f"{base_name}.pth"
        if not os.path.exists(model_path):
            break
        i += 1

    # Salva o modelo
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em {model_path}")

# Dataset e DataLoader
class Word2VecDataset(Dataset):
    def __init__(self, context_pairs, word_to_idx):
        self.context_pairs = context_pairs
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.context_pairs)

    def __getitem__(self, idx):
        word, context_word = self.context_pairs[idx]
        word_idx = self.word_to_idx[word]
        context_idx = self.word_to_idx[context_word]
        return word_idx, context_idx

def prepare_data():
    context_pairs_df = pd.read_csv("/home/jupyter-jphuser04/projects/word2vec/embedding/context_pairs.csv")
    
    if context_pairs_df.isnull().values.any():
        print("Há valores nulos no dataset. Removendo...")
        context_pairs_df = context_pairs_df.dropna()

    if context_pairs_df.empty:
        raise ValueError("O dataset está vazio após a remoção de valores nulos.")

    context_pairs = list(zip(context_pairs_df['Word'], context_pairs_df['Context']))
    vocab = set(context_pairs_df['Word']).union(set(context_pairs_df['Context']))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    dataset = Word2VecDataset(context_pairs, word_to_idx)
    data_loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    return data_loader, len(vocab), word_to_idx

class Word2VecTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(Word2VecTransformerModel, self).__init__()

        # Embeddings Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=1024,
            dropout=dropout,
            activation='gelu',
            batch_first=True 
        )
        
        # Transformer Encoder com múltiplas camadas
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Camada de saída linear
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, word_idx):
        # Passa os índices de palavras pela camada de embeddings
        embedding = self.embeddings(word_idx)

        # O Transformer agora espera [batch size, sequence length, embedding dim]
        transformer_output = self.transformer_encoder(embedding)

        # Transformação linear final
        output = self.fc(transformer_output)

        return output

def train_word2vec(data_loader, vocab_size, word_to_idx, num_heads=4, num_layers=4, lr=0.005, dropout=0.1, epochs=20, device="cuda"):
    model = Word2VecTransformerModel(
        vocab_size=vocab_size,
        embedding_dim=512,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()

    # Diretório para salvar os modelos de cada época
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_silhouette = []
        
        for word_idx, context_idx in data_loader:
            word_idx = word_idx.to(device)
            context_idx = context_idx.to(device)

            optimizer.zero_grad()
            output = model(word_idx)
            loss = criterion(output, model(context_idx), torch.ones(word_idx.shape[0]).to(device))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # Calculando a métrica de silhouette para a época
            embeddings = output.detach().cpu().numpy()
            silhouette_avg = silhouette_score(embeddings, word_idx.cpu().numpy())
            epoch_silhouette.append(silhouette_avg)
        
        avg_loss = epoch_loss / len(data_loader)
        avg_silhouette = np.mean(epoch_silhouette)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Silhouette: {avg_silhouette:.4f}")

        # Salvando o modelo para a época atual
        model_path = f"models/word2vec_light2_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")

    print("Treinamento completo.")


# Executar treinamento
data_loader, vocab_size, word_to_idx = prepare_data()
train_word2vec(data_loader, vocab_size, word_to_idx)
