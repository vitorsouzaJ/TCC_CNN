import os
import random
import shutil

# Definir os diretórios de origem e destino
diretorio_origem = './Banco_de_imagens_original'
diretorio_destino = './data_set_orquideas'

# Criar as pastas de teste, validação e treinamento
diretorio_teste = os.path.join(diretorio_destino, 'test')
diretorio_validacao = os.path.join(diretorio_destino, 'validation')
diretorio_treinamento = os.path.join(diretorio_destino, 'train')
os.makedirs(diretorio_teste, exist_ok=True)
os.makedirs(diretorio_validacao, exist_ok=True)
os.makedirs(diretorio_treinamento, exist_ok=True)

# Definir a porcentagem de imagens para teste, validação e treinamento
porcentagem_teste = 0.15
porcentagem_validacao = 0.15
porcentagem_treinamento = 0.7

# Percorrer todas as pastas dentro do diretório de origem
for nome_pasta in os.listdir(diretorio_origem):
    # Criar as pastas correspondentes nas pastas de teste, validação e treinamento
    pasta_teste = os.path.join(diretorio_teste, nome_pasta)
    pasta_validacao = os.path.join(diretorio_validacao, nome_pasta)
    pasta_treinamento = os.path.join(diretorio_treinamento, nome_pasta)
    os.makedirs(pasta_teste, exist_ok=True)
    os.makedirs(pasta_validacao, exist_ok=True)
    os.makedirs(pasta_treinamento, exist_ok=True)

    # Listar todas as imagens da pasta atual
    imagens = os.listdir(os.path.join(diretorio_origem, nome_pasta))

    # Embaralhar a lista de imagens aleatoriamente
    random.shuffle(imagens)

    # Calcular o número de imagens para teste, validação e treinamento
    total_imagens = len(imagens)
    num_imagens_teste = int(porcentagem_teste * total_imagens)
    num_imagens_validacao = int(porcentagem_validacao * total_imagens)
    num_imagens_treinamento = total_imagens - num_imagens_teste - num_imagens_validacao

    # Separar as imagens de teste
    imagens_teste = imagens[:num_imagens_teste]
    
    # Separar as imagens de validação
    imagens_validacao = imagens[num_imagens_teste:num_imagens_teste + num_imagens_validacao]

    # Separar as imagens de treinamento
    imagens_treinamento = imagens[num_imagens_teste + num_imagens_validacao:]

    # Copiar as imagens de teste para a pasta de teste
    for imagem in imagens_teste:
        origem = os.path.join(diretorio_origem, nome_pasta, imagem)
        destino_teste = os.path.join(pasta_teste, imagem)
        shutil.copyfile(origem, destino_teste)

     # Copiar as imagens de validação para a pasta de validação
    for imagem in imagens_validacao:
        origem = os.path.join(diretorio_origem, nome_pasta, imagem)
        destino_validacao = os.path.join(pasta_validacao, imagem)
        shutil.copyfile(origem, destino_validacao)
        
    # Copiar as imagens de treinamento para a pasta de treinamento
    for imagem in imagens_treinamento:
        origem = os.path.join(diretorio_origem, nome_pasta, imagem)
        destino_treinamento = os.path.join(pasta_treinamento, imagem)
        shutil.copyfile(origem, destino_treinamento)

print('Processamento concluído.')
