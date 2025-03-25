# --------------------------------------------------------------------------------------------------
# import librairies 
# --------------------------------------------------------------------------------------------------
import numpy as np
from osgeo import gdal
import scipy as sp

# --------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------
def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def add_one_obs(past_matrix, new_past_line_vector, new_value):
    """ A function that adds one observation to a covariance matrix, incorporating information from a new image.
            Inputs:
                * past_matrix = covariance matrix of the past images
                * new_past_line_vector = vector of the covariance betwwen the past images and the new one
                * new_value = a scalar representing the variance of the new image

            Outputs:
                * the merged covariance matrix """
    p = past_matrix.shape[0]
    C_tilde = np.zeros((p+1, p+1), dtype=np.complex128)
    C_tilde[0:p, 0:p] = past_matrix
    C_tilde[p, 0:p] = new_past_line_vector
    C_tilde[0:p, p] = new_past_line_vector.conj().T
    C_tilde[p, p] = new_value
    return C_tilde

def calculateMSE(phasedifference, deltathetasim,n_MC,vecL):
    """ A function to calculate the MSE of the phase difernces
    Inputs : 
        * phasedifference : vector of estimated difference
        * deltathetasim : vector of true values of phase differences
    Outputs : 
        * Vector of MSE (size = size of the vector n_samples) """
    return np.array([np.sum(abs(list(phasedifference)[L] - deltathetasim), axis = 0) /n_MC for L in range(len(vecL))])


def read_image(file):
    """
    A function that reads the image from the specified file using GDAL.

    Parameters:
    - file: The path to the image file.

    Returns:
    - image: The image data as a NumPy array.
    """
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    image = band.ReadAsArray()
    return image

def writetoENVIformat(filename, array): 
    """
    Writes the array data to an ENVI format file using GDAL.

    Parameters:
    - filename: The name of the output file.
    - array: The data to be written.

    Returns:
    - None
    """  
    cols = array.shape[1]
    rows = array.shape[0]
    driver = gdal.GetDriverByName('ENVI')
    outRaster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.FlushCache()