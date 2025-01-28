import os
import json

def find_rst_files(directory):
    rst_files = []
    # Percorre todo o diretório e subdiretórios
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".rst"):
                # Adiciona o caminho absoluto do arquivo encontrado
                full_path = os.path.abspath(os.path.join(root, file))
                rst_files.append(full_path)
    return rst_files

def save_to_json(data, output_file):
    # Salva a lista de arquivos em formato JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Caminho do repositório do Linux ou de onde estão os arquivos .rst
    repository_path = r"C:\projects\rag\linux"
    # Arquivo de saída JSON
    output_json = 'rst_files.json'

    # Verifica se o diretório existe
    if not os.path.isdir(repository_path):
        print(f"O diretório {repository_path} não existe.")
    else:
        # Encontrar os arquivos .rst
        rst_files_list = find_rst_files(repository_path)

        # Salvar a lista de arquivos no arquivo JSON
        save_to_json(rst_files_list, output_json)

        print(f"Arquivo JSON gerado com sucesso: {output_json}")
