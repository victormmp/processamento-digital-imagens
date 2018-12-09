"""
Sharpening filters (agucamento)
"""
import numpy
import math
import copy

class SharpeningFilter:

    def __init__(self, pixels, mask=3):
        self.pixels = pixels
        self.mask = mask
    
    def evaluate(self):

        # Assert pixels is a matrix
        assert(type(self.pixels) == numpy.ndarray)

        # Assert mask nxn is an even number
        assert(self.mask % 2 != 0)

        margin_size = self.mask // 2
        median = (self.mask // 2 + 1)

        n_line, n_column, _ = self.pixels.shape

        result = copy.deepcopy(self.pixels)

        for line in range(n_line):
            for column in range(n_column):

                line_start = 0 if line - margin_size < 0 else line - margin_size
                line_end = n_line if line + margin_size > n_line else line + margin_size
                column_start = 0 if column - margin_size < 0 else column - margin_size
                column_end = n_column if column + margin_size > n_column else column + margin_size

                result[line, column] = self.laplace(self.pixels[line_start : (line_end + 1), column_start : (column_end + 1)].flatten('C'))
        
        return result

    def laplace(self, pixels):

        if len(pixels) < self.mask**2:
            pixels_list = list(pixels)
            num_zeros = self.mask**2 - len(pixels)
            pixels_list += [0] * num_zeros
            pixels = numpy.array(pixels_list)

        # Number of pixels on each side n x n
        side = int(numpy.sqrt(pixels.shape))

        laplace_mask = self.build_laplace_mask(side)

        return int(sum([pixel *multiplier for pixel, multiplier in zip(pixels, laplace_mask)]))
    

    @staticmethod
    def build_laplace_mask(side):

        # Position of the center pixel
        mean_position = int(numpy.mean(range(side)))

        c = - 1

        # Build position mapper for each pixel
        laplace_matrix = [c * 1 for _ in range(side**2)]

        laplace_matrix[(mean_position+1)*side- mean_position - 1] = - c * (side*side)

        return laplace_matrix
