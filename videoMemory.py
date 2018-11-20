import os 
from PIL import Image


#retorna o numero de pixels da imagem
def nPixels(arquivo, RGB=True):
    im = Image.open(arquivo)
    width, height = im.size
    im.close()
    return width * height * (2*(RGB) + 1)

def bytesPerVideo(pastaDosFrames, extensao='jpg'):
    return sum([ nPixels(os.path.join(pastaDosFrames,arquivo)) for arquivo in os.listdir(pastaDosFrames) if arquivo.endswith(extensao) ])

def gbPerVideo(pastaDosFrames):
    bytes = bytesPerVideo(pastaDosFrames)
    return float(bytes)/(1024**3)

def mbPerVideo(pastaDosFrames):
    bytes = bytesPerVideo(pastaDosFrames)
    return float(bytes)/(1024**2)

#print(gbPerVideo('/home/hugo/Documents/Mestrado/vot2015/bag'))


