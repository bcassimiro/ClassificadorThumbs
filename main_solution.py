import os
import joblib
import argparse
import sys
from image_tools import image_pipe


def predict_class(image_path):
    image = image_pipe('./Imagens para teste/{}'.format(image_path))

    model = joblib.load('./models/svm.sav')

    result = model.predict(image)

    print({'Resultado do classificador:': result[0]})


parser = argparse.ArgumentParser(description='Submeta uma imagem para o API')
parser.add_argument('image', metavar='img', type=str,
                    help='the input image file path')

args = parser.parse_args()

predict_class(parser.parse_args().image)
print("Predicao concluida")
sys.exit(1)