import sys
from PIL import Image
from src.Filters.Suavization import *
from src.Filters.Sharpening import *
import numpy as np

# sys.path.insert(0, r'./')

new_filter = SuavizationFilter()

imagem = Image.open("IMG1.png").convert('L')
width, height = imagem.size

img = np.array(imagem.getdata()).reshape((height, width, 1))

# Filtragem com filtro media mascara 3
# print('Filtragem com filtro media mascara 3')
# mean_out = new_filter.mean(img).evaluate()
# newImage = Image.fromarray(mean_out.astype('uint8').reshape((height, width)), 'L')
# newImageName = 'IMG1_mean_3_out.png'
# newImage.save(newImageName)

# # Filtragem com filtro media mascara 5
# print("Filtragem com filtro media mascara 5")
# mean_out = new_filter.mean(img, mask=5).evaluate()
# newImage = Image.fromarray(mean_out.astype('uint8').reshape((height, width)), 'L')
# newImageName = 'IMG1_mean_5_out.png'
# newImage.save(newImageName)

# # Filtragem com filtro gaussiano mascara 3
# print("Filtragem com filtro gaussiano mascara 3")
# gauss_out = new_filter.gaussian(img).evaluate()
# newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height, width)), 'L')
# newImageName = 'IMG1_gauss_3_out.png'
# newImage.save(newImageName)

# # Filtragem com filtro gaussiano mascara 5
# print("Filtragem com filtro gaussiano mascara 5")
# gauss_out = new_filter.gaussian(img, mask=5).evaluate()
# newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height, width)), 'L')
# newImageName = 'IMG1_gauss_5_out.png'
# newImage.save(newImageName)

# Filtragem com filtro laplace mascara 3
print('Filtragem com filtro laplace mascara 3')
laplace_out = SharpeningFilter(img).evaluate()
newImage = Image.fromarray(laplace_out.astype('uint8').reshape((height, width)), 'L')
newImageName = 'IMG1_laplace_3_out.png'
newImage.save(newImageName)