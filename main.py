import sys
from PIL import Image
from src.Filters.Suavization import *
import numpy as np

# sys.path.insert(0, r'./')

new_filter = SuavizationFilter()

imagem = Image.open("IMG1.png")
width, height = imagem.size

img = np.array(imagem.getdata()).reshape((height, width, 3))

# Filtragem com filtro media
mean_out = new_filter.mean(img).evaluate()
newImage = Image.fromarray(mean_out.astype('uint8'))
newImageName = 'IMG1_mean_out.png'
newImage.save(newImageName)

# Filtragem com filtro gaussiano
gauss_out = new_filter.gaussian(img).evaluate()
newImage = Image.fromarray(gauss_out.astype('uint8'))
newImageName = 'IMG1_gauss_out.png'
newImage.save(newImageName)