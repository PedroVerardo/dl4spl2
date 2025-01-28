import os
import csv
from collections import Counter

# Função para carregar os arquivos pré-processados
def load_preprocessed_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".rst"):  # Confirma se o arquivo é .rst
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()  # Quebra o texto em palavras
                yield words  # Usar yield para gerar palavras por arquivo, evitando carregar tudo na memória de uma vez

# Criar o vocabulário progressivamente
def build_vocabulary(folder_path):
    vocab_counter = Counter()
    for words in load_preprocessed_files(folder_path):
        vocab_counter.update(words)  # Atualizar o contador com as palavras de cada arquivo
    print(f"Total de palavras únicas no vocabulário: {len(vocab_counter)}")
    return vocab_counter

# Função para salvar o vocabulário em um arquivo CSV
def save_vocabulary_to_csv(vocab, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Frequency'])  # Cabeçalhos
        for word, freq in vocab.items():
            writer.writerow([word, freq])  # Escrever palavra e frequência

# Gerar pares de contexto (palavra-alvo, palavra-contexto)
def create_context_pairs(words, window_size=2):
    context_pairs = []
    for i, word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                context_pairs.append((word, words[j]))
    return context_pairs

# Função para salvar os pares de contexto em um arquivo CSV
def save_context_pairs_to_csv(context_pairs, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Escrever cabeçalhos
        writer.writerow(['Word', 'Context'])
        # Escrever os pares palavra-alvo, palavra-contexto
        for word, context in context_pairs:
            writer.writerow([word, context])

# Carregar os arquivos pré-processados e gerar o vocabulário e os pares de contexto
if __name__ == "__main__":
    folder_path = "./pre-processada"  # Pasta onde estão os arquivos pré-processados (alterar dependendo do seu diretório)

    # Construir o vocabulário
    vocab = build_vocabulary(folder_path)

    # Salvar o vocabulário em um arquivo CSV
    vocab_file = "vocabulary.csv"
    save_vocabulary_to_csv(vocab, vocab_file)
    print(f"Vocabulário salvo em {vocab_file}")

    # Gerar e salvar os pares de contexto
    all_context_pairs = []
    for words in load_preprocessed_files(folder_path):
        context_pairs = create_context_pairs(words, window_size=2)
        all_context_pairs.extend(context_pairs)  # Acumular os pares de contexto de todos os arquivos

    output_file = "context_pairs.csv"
    save_context_pairs_to_csv(all_context_pairs, output_file)
    print(f"Pares de contexto salvos em {output_file}")
