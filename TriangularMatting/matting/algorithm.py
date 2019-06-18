## CSC320 Winter 2017
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import cv2 as cv
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# If you wish to import any additional modules
# or define other utility functions,
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing
    # algorithms. These images are initialized to None and populated/accessed by
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage

    def mattingInput(self):
        return {
            'backA': {'msg': 'Image filename for Background A Color', 'default': None},
            'backB': {'msg': 'Image filename for Background B Color', 'default': None},
            'compA': {'msg': 'Image filename for Composite A Color', 'default': None},
            'compB': {'msg': 'Image filename for Composite B Color', 'default': None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut': {'msg': 'Image filename for Object Color', 'default': ['color.tif']},
            'alphaOut': {'msg': 'Image filename for Object Alpha', 'default': ['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn': {'msg': 'Image filename for Object Color', 'default': None},
            'alphaIn': {'msg': 'Image filename for Object Alpha', 'default': None},
            'backIn': {'msg': 'Image filename for Background Color', 'default': None},
        }

    def compositingOutput(self):
        return {
            'compOut': {'msg': 'Image filename for Composite Color', 'default': ['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.

    def useTriangulationResults(self):

        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the
    # matting instance's private dictionary object. The key
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Reading image failed'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        img = cv.imread(fileName)
        if img is not None:
            success = True
            self._images[key] = img
            msg = 'Reading image succeed'

        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63.
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Writing image failed'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if self._images[key] is not None:
            cv.imwrite(fileName, self._images[key])
            success = True
            msg = 'Writing image succeed'

        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary
    # ojbect.
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)

        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Matting failed'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        back_a = self._images['backA']
        back_b = self._images['backB']
        comp_a = self._images['compA']
        comp_b = self._images['compB']
        if (back_a is not None) and (
                    back_b is not None) and (
                    comp_a is not None) and (
                    comp_b is not None):
            success = True
            msg = 'Matting succeed'
            # change the data type of every image
            back_a = back_a.astype(np.float16)/255.0
            back_b = back_b.astype(np.float16)/255.0
            comp_a = comp_a.astype(np.float16)/255.0
            comp_b = comp_b.astype(np.float16)/255.0
            # stack two background images matrices; also
            # stack two composite images matrices together
            back = np.dstack((back_a, back_b))
            comp = np.dstack((comp_a, comp_b))
            c_delta = comp - back
            x, y, z = c_delta.shape
            b_x, b_y, b_z = back.shape
            # make background matrix be column vector for calculation
            back_reshaped = np.reshape(back, (b_x, b_y, b_z, 1))
            # make delta matrix be column vector for calculation
            c_delta_reshaped = np.reshape(c_delta, (x, y, z, 1))

            # make coefficient matrix with the following 4 lines
            temp1 = np.full((x, y, z, 4), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                           [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            temp_back = np.zeros((x, y, z, 4))
            temp_back[:x, :y, :z, 3:4] = back_reshaped
            coefficient = temp1 - temp_back

            # calculate pseudo inverse of this coefficient matrix
            inverted = Matting.pinv(coefficient)
            # solve the equation, clip results
            result = np.clip(np.matmul(inverted, c_delta_reshaped), 0.0, 1.0)
            # change result back to row vectors
            result_trans = np.reshape(result, (x, y, 4))
            # save result to alpha_out and col_out respectively
            alpha_out = (result_trans[:, :, 3:4] * 255).astype(np.uint8)
            col_out = (result_trans[:, :, :3] * 255).astype(np.uint8)
            self._images['alphaOut'] = alpha_out
            self._images['colOut'] = col_out

            # Doing triangular matting algorithm here
        #########################################

        return success, msg

    def createComposite(self):
        """
success, errorMessage = createComposite(self)

        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Composition failed'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        back_in = self._images['backIn']
        col_in = self._images['colIn']
        alpha_in = self._images['alphaIn']
        if (back_in is not None) and (col_in is not None) and (alpha_in is not None):
            success = True
            msg = 'Composition succeed'
            # Matting equation C = C0 + (1 - alpha) * Ck
            # We need to multiply 1/255 to alpha value first to convert it back to value
            # between 0 and 1
            comp_out = col_in + (1 - (1/float(255))*alpha_in) * back_in
            self._images['compOut'] = comp_out

        #########################################

        return success, msg

    # Helper static method to inverse every sub-matrix of a 4D matrix
    # then inverse them
    @staticmethod
    def diagonal_inverse(a):
        r, c, index = a.shape
        matrix = np.zeros((r, c, index, index))
        reciprocals = 1 / a
        row, column = np.diag_indices(index)
        matrix[:, :, row, column] = reciprocals
        return matrix

    # Helper static method to get the pseudo-inverse of every sub-matrix
    # of a 4D matrix using SVD
    @staticmethod
    def pinv(a, rcond=1e-15):
        swap = np.arange(a.ndim)
        swap[[-2, -1]] = swap[[-1, -2]]
        u, s, v = np.linalg.svd(a, full_matrices=False)
        s_inverse = Matting.diagonal_inverse(s)
        u_transpose = np.transpose(u, swap)
        v_transpose = np.transpose(v, swap)
        first = np.matmul(s_inverse, u_transpose)
        return np.matmul(v_transpose, first)
