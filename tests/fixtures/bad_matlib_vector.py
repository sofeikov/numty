import numpy as np
import numpy.matlib

# numpy.matlib.zeros builds a 2-D matrix, not np.zeros' 1-D array.
v: np.ndarray[tuple[int], np.dtype[np.float64]] = numpy.matlib.zeros(3)
