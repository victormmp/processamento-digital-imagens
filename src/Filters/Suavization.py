"""
FUNCTIONS MODULE
"""

import numpy
import math
import copy

class SuavizationFilter:
    """Filter class. Implements suavization filters"""

    # Weights for each parameter. Define a different weight vector
    # to modify weightened sum
    weights = None
    func = None
    mask = 3

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = weights

    def set_weights(self, weights):
        """
        Set the weights vector.

        This method is build to allow chain implementation, such as:

        filter = SuavizationFilter()
        weights = [...]
        result = filter.set_weights(weights).mean(pixels)
        
        """

        self.weights = weights
        return self

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

                result[line, column] = self.func(self.pixels[line_start : (line_end + 1), column_start : (column_end + 1)].flatten('C'))
        
        return result


    def mean(self, pixels, mask=3):
        """Mean suavization filter."""

        self.func = lambda x: int(numpy.mean(x))
        self.pixels = pixels
        self.mask = mask

        return self


    def weightened_mean(self, pixels, mask=3):
        """Weightened mean suavization filter."""

        self.pixels = pixels
        self.mask = mask

        if self.weights is None:
            self.weights = self.generate_unit_weights(self.pixels.size)
        
        self.func = lambda x: sum([pixel * weight for pixel, weight in zip(x, self.weights)]) / sum(self.weights)

        return self
    
    def gaussian(self, pixels, mask=3):
        """Gaussian filter"""

        self.pixels = pixels
        self.mask = mask
        self.func = self.gaussian_func

        return self


    def gaussian_func(self, pixels):

        # Number of pixels on each side n x n
        side = int(numpy.sqrt(pixels.shape))

        distances = self.build_distance_matrix(side)        
        variance = numpy.var(pixels)

        gaussian_matrix = [1 / (2 * math.pi * variance) * math.pow(math.e, (distance / (2 * variance))) 
                           for distance in distances]

        return int(numpy.mean(gaussian_matrix))


    @staticmethod
    def build_distance_matrix(side):

        # Position of the center pixel
        mean_position = int(numpy.mean(range(side)))

        # Build position mapper for each pixel
        position_matrix = []
        for line in range(side):
            for column in range(side):
                position_matrix.append([line - mean_position, column - mean_position])
        
        # Build squared distance matrix
        distance = [x**2 + y**2 for x, y in position_matrix]

        return distance


    @staticmethod
    def generate_unit_weights(size):
        """Generate a vector of '1's"""
        return [1 for _ in range(size)]