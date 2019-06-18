## CSC320 Winter 2017
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################
import scipy.ndimage as sc
# If you need to import any additional packages
# place them here. Note that the reference
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1],
#         you should assume confidences are represented as variables of type
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace this dummy value with your own code
    # Get the numpy array of size (2w + 1)x(2w + 1) from confidenceImage.
    confidence, _ = copyutils.getWindow(confidenceImage, psiHatP._coords, psiHatP._w)
    # Get the numpy array of size (2w + 1)x(2w + 1) from the filledImage to see which
    # pixel has been filled.
    filled, valid = copyutils.getWindow(filledImage, psiHatP._coords, psiHatP._w)
    # Get the total number of pixels which are valid (inbound)
    total = np.sum(valid)
    # Sum up for the total confidence value from those pixels which are filled, then
    # divided by the total number of inbound pixels to get the mean confidence value.
    C = np.sum(confidence[filled != 0]) / total
    #########################################
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the
#             x axis (ie. the horizontal axis).
#


def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace these dummy values with your own code
    kernel_size = 3
    inpainted, _ = copyutils.getWindow(inpaintedImage, psiHatP._coords, psiHatP._w)
    # Change patch from color to gray scale
    inpainted = cv.cvtColor(inpainted, cv.COLOR_BGR2GRAY)
    filled, _ = copyutils.getWindow(filledImage, psiHatP._coords, psiHatP._w)
    # Try to erode the filled patch used cv.erode() with 3x3 kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv.erode(filled, kernel, iterations=1, borderType=cv.BORDER_REPLICATE)
    # Compute gradient along x axis
    gx = cv.Scharr(src=inpainted, ddepth=cv.CV_32F, dx=1, dy=0, borderType=cv.BORDER_REPLICATE)
    # Compute gradient along y axis
    gy = cv.Scharr(src=inpainted, ddepth=cv.CV_32F, dx=0, dy=1, borderType=cv.BORDER_REPLICATE)
    # Filter out pixels that are unfilled
    gx *= erosion > 0
    gy *= erosion > 0
    # Compute the magnitude of gradient of every pixel
    gradient = np.sqrt(gx*gx + gy*gy)
    # Find the x and y coordinates of the maximum gradient magnitude from gradient matrix
    x, y = np.unravel_index(gradient.argmax(), gradient.shape)
    Dx = -gy[x][y]
    Dy = gx[x][y]
    #########################################

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#


def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace these dummy values with your own code
    # Get target numpy array of size (2w + 1)x(2w + 1) from fillFront image
    target, _ = copyutils.getWindow(fillFront, psiHatP._coords, psiHatP._w)
    # Use 2D gaussian kernel to smooth and filter target patch of fillFront
    gaussian_filter = sc.filters.gaussian_filter(target, 1)
    if target.size == 1:  # The case that when there is only one pixel in target patch
        Nx = None
        Ny = None
    else:
        # Compute the gradient of the patch, get the center gradient of the patch
        nx = cv.Scharr(src=gaussian_filter, ddepth=cv.CV_32F, dx=1, dy=0,
                       borderType=cv.BORDER_REPLICATE)[psiHatP._w][psiHatP._w]
        ny = cv.Scharr(src=gaussian_filter, ddepth=cv.CV_32F, dx=0, dy=1,
                       borderType=cv.BORDER_REPLICATE)[psiHatP._w][psiHatP._w]
        # Change to unit vector
        d = np.sqrt(ny*ny + nx*nx)
        if d != 0:
            nx /= d
            ny /= d
        Ny = ny
        Nx = nx
    #########################################

    return Ny, Nx
