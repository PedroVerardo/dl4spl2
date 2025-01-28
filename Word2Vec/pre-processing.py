import os
import json
import re
import nltk
from nltk.corpus import stopwords

# Baixar stopwords da biblioteca nltk (execute isso uma vez)
nltk.download('stopwords')

# Criar pasta pre-processada se não existir
if not os.path.exists('./pre-processada'):
    os.makedirs('./pre-processada')

# Função para carregar arquivos rst_files.json
def load_rst_files(json_path):
    with open(json_path, 'r') as f:
        file_paths = json.load(f)
    return file_paths

# Função de pré-processamento do texto
def preprocess_text(text):
    # Converter para minúsculas
    text = text.lower()

    # Remover números
    text = re.sub(r'\d+', '', text)

    # Remover pontuações
    text = re.sub(r'[^\w\s]', '', text)

    # Remover caracteres não-ASCII (ou seja, manter apenas caracteres em inglês)
    text = ''.join([char for char in text if ord(char) < 128])

    # Remover múltiplos espaços
    text = re.sub(r'\s+', ' ', text).strip()

    # Remover stopwords (palavras inúteis)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    # Juntar as palavras filtradas novamente em uma string
    clean_text = ' '.join(filtered_words)

    return clean_text

# Função para carregar e pré-processar os arquivos
def load_and_preprocess_files(file_paths):
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Pré-processar o texto
            clean_text = preprocess_text(text)
            
            # Definir novo caminho de arquivo
            new_filename = os.path.basename(path)
            new_path = os.path.join('./pre-processada', new_filename)
            
            # Salvar o arquivo pré-processado
            with open(new_path, 'w', encoding='utf-8') as new_file:
                new_file.write(clean_text)
            print(f"Arquivo {new_filename} salvo em './pre-processada/'")

# Caminho para o arquivo rst_files.json
json_path = "rst_files.json"
file_paths = load_rst_files(json_path)

# Pré-processar os arquivos e salvá-los
load_and_preprocess_files(file_paths)
