import sys
from PIL import Image
from src.Filters.Suavization import *
from src.HistEqualizer.HistogramEqualization import *
import numpy as np

# sys.path.insert(0, r'./')

print("Iniciando Script")

new_filter = SuavizationFilter()

path_out_imgs = "./out_imgs/"

imagem = Image.open("IMG1.png").convert('L')
imagem2 = Image.open("IMG2.png").convert('L')
width, height = imagem.size
width2, height2 = imagem2.size

colorDepth = 8

img = np.array(imagem.getdata()).reshape((height, width, 1))
img2 = np.array(imagem2.getdata()).reshape((height2, width2, 1))



# Filtragem com filtro media mascara 3 - IMG1
print('Filtragem com filtro media mascara 3 - IMG1')
mean_out = new_filter.mean(img).evaluate()
newImage = Image.fromarray(mean_out.astype('uint8').reshape((height, width)), 'L')
newImageName = path_out_imgs + "IMG1/" + 'IMG1_mean_3_out.png'
newImage.save(newImageName)

# Filtragem com filtro media mascara 3 - IMG2
print('Filtragem com filtro media mascara 3 - IMG2')
mean_out = new_filter.mean(img2).evaluate()
newImage = Image.fromarray(mean_out.astype('uint8').reshape((height2, width2)), 'L')
newImageName = path_out_imgs + "IMG2/" + 'IMG2_mean_3_out.png'
newImage.save(newImageName)

# Filtragem com filtro media mascara 5 - IMG1
print("Filtragem com filtro media mascara 5 - IMG1")
mean_out = new_filter.mean(img, mask=5).evaluate()
newImage = Image.fromarray(mean_out.astype('uint8').reshape((height, width)), 'L')
newImageName = path_out_imgs + "IMG1/" + 'IMG1_mean_5_out.png'
newImage.save(newImageName)

# Filtragem com filtro media mascara 5 - IMG2
print("Filtragem com filtro media mascara 5 - IMG2")
mean_out = new_filter.mean(img2, mask=5).evaluate()
newImage = Image.fromarray(mean_out.astype('uint8').reshape((height2, width2)), 'L')
newImageName = path_out_imgs + "IMG2/" + 'IMG2_mean_5_out.png'
newImage.save(newImageName)

# Filtragem com filtro gaussiano mascara 3 - IMG1
print("Filtragem com filtro gaussiano mascara 3 - IMG1")
gauss_out = new_filter.gaussian(img).evaluate()
newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height, width)), 'L')
newImageName = path_out_imgs + "IMG1/" + 'IMG1_gauss_3_out.png'
newImage.save(newImageName)

# Filtragem com filtro gaussiano mascara 3 - IMG2
print("Filtragem com filtro gaussiano mascara 3 - IMG2")
gauss_out = new_filter.gaussian(img2).evaluate()
newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height2, width2)), 'L')
newImageName = path_out_imgs + "IMG2/" + 'IMG2_gauss_3_out.png'
newImage.save(newImageName)

# Filtragem com filtro gaussiano mascara 5 - IMG1
print("Filtragem com filtro gaussiano mascara 5 - IMG1")
gauss_out = new_filter.gaussian(img, mask=5).evaluate()
newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height, width)), 'L')
newImageName = path_out_imgs + "IMG1/" + 'IMG1_gauss_5_out.png'
newImage.save(newImageName)

# Filtragem com filtro gaussiano mascara 5 - IMG2
print("Filtragem com filtro gaussiano mascara 5 - IMG2")
gauss_out = new_filter.gaussian(img2, mask=5).evaluate()
newImage = Image.fromarray(gauss_out.astype('uint8').reshape((height2, width2)), 'L')
newImageName = path_out_imgs + "IMG2/" + 'IMG2_gauss_5_out.png'
newImage.save(newImageName)

#Equalização de Histograma - IMG1
print("Equalização de Histograma - IMG1")
imgName = path_out_imgs + "IMG1/" + "IMG1"
hist_eq1 = HistogramEqualization(img,colorDepth,imgName)
img_out = hist_eq1.evaluate()
newImage = Image.fromarray(img_out.astype('uint8').reshape((height, width)), 'L')
newImageName = path_out_imgs + "IMG1/" + 'IMG1_hist_eq_out.png'
newImage.save(newImageName)

#Equalização de Histograma - IMG2
print("Equalização de Histograma - IMG2")
imgName = path_out_imgs + "IMG2/" + "IMG2"
hist_eq2 = HistogramEqualization(img2,colorDepth,imgName)
img_out = hist_eq2.evaluate()
newImage = Image.fromarray(img_out.astype('uint8').reshape((height2, width2)), 'L')
newImageName = path_out_imgs + "IMG2/" + 'IMG2_hist_eq_out.png'
newImage.save(newImageName)

print("Fim")