"""
Histogram Equalization Class
"""

import numpy
import math
import copy
import matplotlib.pyplot as plt

class HistogramEqualization:
    """Implements Histogram Equalization"""

    imgName = "IMG" #ImageName
    colorDepth = 8 #Intensity represented by 8 bits

    def __init__(self, pixels, colorDepth=8, imgName="IMG"):
        self.pixels = pixels
        self.colorDepth = colorDepth
        self.imgName = imgName

    def evaluate(self):

        # Assert pixels is a matrix
        assert(type(self.pixels) == numpy.ndarray)
        
        height, width, _ = self.pixels.shape
        img = self.pixels.reshape(height*width)
        
        L = 2**self.colorDepth
        
        # Assert color depth is coherent
        assert(L > numpy.amax(img))

        # Calculation of intesity frequencies 
        frequency = numpy.zeros(L)
        for pixel in img:
            frequency[pixel] += 1/(width*height)
        
        # Print histogram of original image
        fig_name = self.imgName + "_hist"
        self.printHistogram(frequency,fig_name)
        
        # Creation of intensity transformation function
        eq_transformation_func = numpy.zeros(L)
        for intesity in range(L):
            sum_previous = 0
            for previous in range(intesity):
                sum_previous =+ eq_transformation_func[previous]
            eq_transformation_func[intesity] = sum_previous + (L-1) * frequency[intesity] 

        eq_transformation_func = numpy.around(eq_transformation_func) # Round new intensity values 
        eq_transformation_func = eq_transformation_func.astype(int) # Transform to integer
        
        # Generation of equalized image from the transformation function
        eq_img = eq_transformation_func[img]

        # Calculation of equalized intesity frequencies 
        frequency_eq = numpy.zeros(L)
        for pixel in eq_img:
            frequency_eq[pixel] += 1/(width*height)
        
        # Print histogram of equalized image
        fig_name = self.imgName + "_hist_eq"
        self.printHistogram(frequency_eq,fig_name)
        
        result = numpy.array(eq_img).reshape((height, width, 1))
        
        return result


    @staticmethod
    def printHistogram(frequency, figName):
        f = plt.figure()    
        plt.bar(range(len(frequency)),frequency)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        figName = figName + ".pdf" 
        f.savefig(figName, bbox_inches='tight')
