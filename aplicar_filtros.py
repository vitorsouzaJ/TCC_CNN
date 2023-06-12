import cv2
import glob
import os

# Diretório contendo as imagens originais
diretorio_origem = './data_set_orquideas'

# Percorrer todas as imagens nos diretórios de origem e subdiretórios recursivamente
for caminho_origem in glob.glob(os.path.join(diretorio_origem, '**/*.*'), recursive=True):
    # Verificar se é um arquivo de imagem
    if caminho_origem.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Carregar a imagem original
        imagem_original = cv2.imread(caminho_origem)

        # Converter a imagem para escala de cinza
        imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro de detecção de bordas para obter o efeito de desenho a lápis
        imagem_bordas = cv2.Canny(imagem_cinza, threshold1=30, threshold2=100)
        imagem_bordas = cv2.bitwise_not(imagem_bordas)
        imagem_bordas = cv2.cvtColor(imagem_bordas, cv2.COLOR_GRAY2BGR)
        imagem_desenho = cv2.addWeighted(imagem_original, 0.5, imagem_bordas, 0.5, 0.0)

        # Rotacionar a imagem original em 90 graus no sentido horário
        imagem_rotacao_90 = cv2.rotate(imagem_original, cv2.ROTATE_90_CLOCKWISE)

        # Rotacionar a imagem em escala de cinza em 90 graus no sentido horário
        imagem_cinza_rotacao_90 = cv2.rotate(imagem_cinza, cv2.ROTATE_90_CLOCKWISE)

        # Rotacionar a imagem com filtro de desenho a lápis em 90 graus no sentido horário
        imagem_desenho_rotacao_90 = cv2.rotate(imagem_desenho, cv2.ROTATE_90_CLOCKWISE)
        
        # Rotacionar a imagem original em 180 graus no sentido horário
        imagem_rotacao_180 = cv2.rotate(imagem_original, cv2.ROTATE_180)

        # Rotacionar a imagem em escala de cinza em 180 graus no sentido horário
        imagem_cinza_rotacao_180 = cv2.rotate(imagem_cinza, cv2.ROTATE_180)

        # Rotacionar a imagem com filtro de desenho a lápis em 180 graus no sentido horário
        imagem_desenho_rotacao_180 = cv2.rotate(imagem_desenho, cv2.ROTATE_180)
        
        # Rotacionar a imagem original em 270 graus no sentido horário
        imagem_rotacao_270 = cv2.rotate(imagem_original, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Rotacionar a imagem em escala de cinza em 270 graus no sentido horário
        imagem_cinza_rotacao_270 = cv2.rotate(imagem_cinza, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Rotacionar a imagem com filtro de desenho a lápis em 270 graus no sentido horário
        imagem_desenho_rotacao_270 = cv2.rotate(imagem_desenho, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Obter o diretório e o nome do arquivo da imagem original
        diretorio_origem, nome_arquivo = os.path.split(caminho_origem)

        # Caminhos completos dos arquivos de destino na mesma pasta do diretório de origem
        caminho_destino_cinza = os.path.join(diretorio_origem, 'cinza_' + nome_arquivo)
        caminho_destino_desenho = os.path.join(diretorio_origem, 'desenho_' + nome_arquivo)
        caminho_destino_rotacao_90 = os.path.join(diretorio_origem, 'rotacao_90_' + nome_arquivo)
        caminho_destino_cinza_rotacao_90 = os.path.join(diretorio_origem, 'cinza_rotacao_90_' + nome_arquivo)
        caminho_destino_desenho_rotacao_90 = os.path.join(diretorio_origem, 'desenho_rotacao_90_' + nome_arquivo)
        caminho_destino_rotacao_180 = os.path.join(diretorio_origem, 'rotacao_180_' + nome_arquivo)
        caminho_destino_cinza_rotacao_180 = os.path.join(diretorio_origem, 'cinza_rotacao_180_' + nome_arquivo)
        caminho_destino_desenho_rotacao_180 = os.path.join(diretorio_origem, 'desenho_rotacao_180_' + nome_arquivo)
        caminho_destino_rotacao_270 = os.path.join(diretorio_origem, 'rotacao_270_' + nome_arquivo)
        caminho_destino_cinza_rotacao_270 = os.path.join(diretorio_origem, 'cinza_rotacao_270_' + nome_arquivo)
        caminho_destino_desenho_rotacao_270 = os.path.join(diretorio_origem, 'desenho_rotacao_270_' + nome_arquivo)

        # Salvar as imagens nos diretórios de destino
        cv2.imwrite(caminho_destino_cinza, imagem_cinza)
        cv2.imwrite(caminho_destino_desenho, imagem_desenho)
        cv2.imwrite(caminho_destino_rotacao_90, imagem_rotacao_90)
        cv2.imwrite(caminho_destino_cinza_rotacao_90, imagem_cinza_rotacao_90)
        cv2.imwrite(caminho_destino_desenho_rotacao_90, imagem_desenho_rotacao_90)
        cv2.imwrite(caminho_destino_rotacao_180, imagem_rotacao_180)
        cv2.imwrite(caminho_destino_cinza_rotacao_180, imagem_cinza_rotacao_180)
        cv2.imwrite(caminho_destino_desenho_rotacao_180, imagem_desenho_rotacao_180)
        cv2.imwrite(caminho_destino_rotacao_270, imagem_rotacao_270)
        cv2.imwrite(caminho_destino_cinza_rotacao_270, imagem_cinza_rotacao_270)
        cv2.imwrite(caminho_destino_desenho_rotacao_270, imagem_desenho_rotacao_270)
print('Processamento concluído.')
