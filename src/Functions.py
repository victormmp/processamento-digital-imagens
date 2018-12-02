"""
FUNCTIONS MODULE
"""

import numpy
import math

class SuavizationFilters:
    """Filter class. Implements suavization filters"""

    # Weights for each parameter. Define a different weight vector
    # to modify weightened sum
    weights = None

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = weights
    

    def mean(self, pixels):
        """Mean suavization filter."""

        return int(numpy.mean(pixels))
    

    def weightened_mean(self, pixels):
        """Weightened mean suavization filter."""

        if self.weights is not None:
            weights = self.weights
        else:
            weights = self.generate_unit_weights(len(pixels))
        
        pixel_sum = sum([pixel * weight for pixel, weight in zip(pixels, weights)])
        weight_sum = sum(weights)

        return pixel_sum / weight_sum
    

    def gaussian(self, pixels):

        # Number of pixels on each side n x n
        side = int(numpy.sqrt(pixels))

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
        

    def set_weights(self, weights):
        """Set the weights vector"""

        self.weights = weights
        return self


    @staticmethod
    def generate_unit_weights(size):
        """Generate a vector of '1's"""
        return [1 for _ in range(size)]