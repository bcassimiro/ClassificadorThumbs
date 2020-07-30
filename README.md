# Classificador de imagens "*Thumbs Up*" e "*Thumbs Down*"
Autor: Bernardo Cassimiro Fonseca de Oliveira

Essa ferramenta classifica imagens com "*Thumbs Up*" e "*Thumbs Down*" utilizando uma Máquina de Vetor Suporte (SVM, do inglês *Support Vector Machine*) com pré-processamento usando uma Histograma de Gradientes (HOG, do inglês *Histogram of Gradients*).

O dataset utilizado para o treinamento desse classificador foi criado por mim mesmo, filmando minhas mãos com os sinais de positivo e negativo em diferentes ângulos com o celular. Os frames desses vídeos foram extraídos, gerando 1400 imagens para o treinamento e 800 imagens para teste. Além disso, 51 imagens retiradas da internet foram utilizadas para testar a capacidade de extrapolação do classificador.

Neste repositório, existem as seguintes pastas/arquivos:
- Pasta "generalization" contendo as imagens usadas para a extrapolação do classificador.
- Pasta "Imagens para teste" contendo as vinte imagens requisitadas para o teste da ferramenta.
- Pasta "models" contendo o modelo do arquivo .py que faz a classificação.
- Pasta "test" contendo as imagens usadas para o teste do classificador.
- Pasta "training" contendo as imagens usadas para o treinamento do classificador.
- Arquivo "generalizationLABELS.csv" contendo as *labels* da extrapolação do classificador.
- Arquivo "testLABELS.csv" contendo as *labels* do teste do classificador.
- Arquivo "trainingLABELS.csv" contendo as *labels* do treinamento do classificador.
- Arquivo "image_tools.py" contendo o arquivo para processamento da imagens da pasta "Imagens para teste".
- Arquivo "main_solution.py" contendo o arquivo principal para rodar o classificador direto do console.
- Arquivo "Relatorio.ipynb" contendo o notebook que explica a ferramenta desenvolvida.
- Arquivo "requirements.txt" contendo as bibliotecas necessárias para rodar a ferramenta.
- Arquivo "SVMwithHOG.py" contendo a ferramenta em si.
- Arquivo "README.md", este arquivo.

As instruções para operar a ferramenta são as seguintes:
1) Clonar esse repositório para a sua interface
2) Acessar a pasta clonada pelo console do Anaconda
3) Criar um ambiente virtual no console do Anaconda utilizando o comando ```conda create --name env_name --file requirements.txt```
4) Esperar a instalação das bibliotecas necessárias
5) Ativar o ambiente virtual com o comando ```conda activate env_name```
6) Digite ```python main_solution.py NOMEdaIMAGEM```, em que "NOMEdaIMAGEM" é o nome de uma das imagens da pasta "Imagens para teste" (ex. 01.bmp)
7) Espere o resultado da classificação e confira se ele de fato é coerente com a imagem testada, sendo o resultado igual a "1" se a imagem for classificada como um "*Thumbs Up*" e igual a "0" se a imagem for classificada como um "*Thumbs Down*"
