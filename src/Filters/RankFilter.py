"""
Rank Filters
"""

import numpy
import math
import copy

class RankFilter:
    """Filter class. Implements rank filters"""

    # Weights for each parameter. Define a different weight vector
    # to modify weightened sum

    func = None
    mask = 3

    def __init__(self):
        pass

    def evaluate(self):
        
        # Assert pixels is a matrix
        assert(type(self.pixels) == numpy.ndarray)

        # Assert mask nxn is an even number
        assert(self.mask % 2 != 0)

        margin_size = self.mask // 2

        n_line, n_column, _ = self.pixels.shape

        result = copy.deepcopy(self.pixels)

        for line in range(n_line):
            for column in range(n_column):

                line_start = 0 if line - margin_size < 0 else line - margin_size
                line_end = n_line if line + margin_size > n_line else line + margin_size
                column_start = 0 if column - margin_size < 0 else column - margin_size
                column_end = n_column if column + margin_size > n_column else column + margin_size

                result[line, column] = self.func(self.pixels[line_start : (line_end + 1),
                                                             column_start : (column_end + 1)].flatten('C'))
        
        return result


    def median(self, pixels, mask=3):
        """Median filter."""

        self.pixels = pixels
        self.mask = mask
        self.func = lambda x: int(numpy.median(x))

        return self

    
    def maximum(self, pixels, mask=3):
        """Maximum filter"""

        self.pixels = pixels
        self.mask = mask
        self.func = lambda x: int(numpy.max(x))

        return self