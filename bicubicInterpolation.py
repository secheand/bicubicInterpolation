import numpy as np
import dippykit as dip
import cv2 as cv
import math

# NOTE: This program uses dippykit package which can be found at https://dippykit.github.io/dippykit/

"""
The function bicubicTransform takes in an image, scaling factors, and a rotation angle, and returns a
resampled image using bicubic interpolation.

:param Lrows: Scaling factor for rows
:param Lcols: Scaling factor for columns
:param theta: Angle of rotation in degrees
:param image: Image that you want to apply the resampling to transformation to
:return: the resampled image.
"""
def bicubicTransform(Lrows,Lcols,theta, image):
    # Get a transform matrix for the resizing and one for the rotation.
    # Multiply them together to obtain the combined transform matrix
    transMatrix = np.array([[Lcols,0],[0,Lrows]])
    theta = np.deg2rad(theta)
    rotMatrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    M = np.matmul(rotMatrix, transMatrix)

    # Chose which subfunction to call depending on what is asked to do. If there is
    # rotation choose the subfunction that allows rotation
    if theta != 0:
        resampledImage = rotateAndInterpolate(image, M, theta, Lrows, Lcols)
    else:
        resampledImage = resizeAndinterpolate(image, Lrows, Lcols)

    return resampledImage

"""
The function resizes and fill empty pixels with bicubic interpolation.

:param image: Image that you want to resize
:param Lrows: Scaling factor for rows
:param Lcols: Scaling factor for columns
:return: the resampled and interpolated image.
"""
def resizeAndinterpolate(image, Lrows, Lcols):
    # find size of resampled image
    imageResampled = np.uint8(np.zeros((round(image.shape[0]*1/Lrows), round(image.shape[1]*1/Lcols))))
    
    # For each pixel in the image
    for m in range(imageResampled.shape[0]):
        for n in range(imageResampled.shape[1]):
            # find the corresponding pixel in the original image
            y = Lcols*(n)
            x = Lrows*(m)

            # if the pixel is an integer and within bound grab its value without interpolation
            if float(x).is_integer and float(y).is_integer:
                if x >= image.shape[0] or x < 0 or y >= image.shape[1] or y < 0:
                    imageResampled[m,n] = 0
                else:
                    imageResampled[m][n] = image[int(x)][int(y)]
            # if not, then interpolate
            else:
                # for each of the 16 neighboors
                for k in range(-1,2):
                    for l in range(-1,2):
                        # Loop through the coordinates of the 16 neighboors
                        xCoord = math.floor(x)+k
                        yCoord = math.floor(y)+l
                        # If within bounds, compute weights and multiply pixel values
                        if xCoord >= image.shape[0] or xCoord < 0 or yCoord >= image.shape[1] or yCoord < 0:
                            imageResampled[m,n] = imageResampled[m][n]+0
                        else:
                            imageResampled[m][n] = imageResampled[m][n] + W(m,k,Lrows)*W(n,l,Lcols)*image[xCoord][yCoord]

    return imageResampled

"""
The function takes an image, a transformation matrix, an angle, and
scaling factors, and returns a resized and rotated version image using bicubic interpolation.

:param image: Image that you want to resize and rotate
:param M: Transformation matrix for rotation and scaling
:param theta: Angle of rotation in radians
:param Lrows: Scaling factor for rows
:param Lcols: Scaling factor for columns
:return: the resized and rotated image.
"""
def rotateAndInterpolate(image, M, theta, Lrows, Lcols):
    # Find the dimensions of the new image using rotation angle and scaling ratios.
    # Add some padding to prevent cropping the image on the edges
    newHeight = (round(abs(image.shape[0]*1/Lrows*math.cos(theta))) + round(abs(image.shape[1]*1/Lcols*math.sin(theta))))+700
    newWidth = (round(abs(image.shape[1]*1/Lcols*math.cos(theta))) + round(abs(image.shape[0]*1/Lrows*math.sin(theta))))+700

    # Find center coordinates for the original and new images
    x0_original = image.shape[0]//2
    y0_original = image.shape[1]//2
    x0_resampled = newHeight//2
    y0_resampled = newWidth//2

    # Create resampled image array
    imageResampled = np.uint8(np.zeros((newHeight, newWidth)))

    # For each pixel in the image
    for m in range(imageResampled.shape[0]):
        for n in range(imageResampled.shape[1]):
            # find the corresponding pixel in the original image using the transform matrix.
            # It accounts for rotation around the center of the image by substracting then center coordinates
            y = M[0][0]*(n-y0_resampled) + M[0][1]*(m-x0_resampled)
            x = M[1][0]*(n-y0_resampled) + M[1][1]*(m-x0_resampled)

            # if the pixel is an integer and within bound grab its value without interpolation
            if float(x).is_integer and float(y).is_integer:
                if x+x0_original >= image.shape[0] or x+x0_original < 0 or y+y0_original >= image.shape[1] or y+y0_original < 0:
                    imageResampled[m,n] = 0
                else:
                    imageResampled[m][n] = image[int(x+x0_original)][int(y+y0_original)]
            # if not, then interpolate
            else:
                # for each of the 16 neighboors
                for k in range(-1,2):
                    for l in range(-1,2):
                        # Loop through the coordinates of the 16 neighboors
                        # account for rotation around center by substracting center coordinates
                        xCoord = math.floor(x)+k+x0_original
                        yCoord = math.floor(y)+l+y0_original

                        # If within bounds, compute weights and multiply pixel values
                        if xCoord >= image.shape[0] or xCoord < 0 or yCoord >= image.shape[1] or yCoord < 0:
                            imageResampled[m,n] = imageResampled[m][n]+0
                        else:
                            imageResampled[m][n] = imageResampled[m][n] + W(m,k,Lrows)*W(n,l,Lcols)*image[xCoord][yCoord]
    return imageResampled

def W(m, k, L):
    """
    The function calculates bicubic interpolation weights based on the input parameters
    
    :param m: Represents coordinate of pixel
    :param k: Represents the index of the neighbooring pixel
    :param L: Represents the scaling ratio
    :return: weight for bicubic interpolation
    """
    delta = m/(1/L) - math.floor(m/(1/L))
    if(k == -1):
        return (-delta**3+2*delta**2-delta) / 2
    elif(k == 0):
        return (3*delta**3-5*delta**2+2) / 2
    elif(k == 1):
        return (-3*delta**3+4*delta**2+delta) / 2
    else:
        return (delta**3-delta**2) / 2

"""
The main function resamples an image, then does the inverse transform and compares it
to the original image. All transformations are done using bicubic interpolation
"""
def main():
    # Load image
    image = cv.imread('barbara.png', cv.IMREAD_GRAYSCALE)
    
    # First transform: Increase rows by 2.5, decrease columns by 1.7, and rotate
    # 27.5º clockwise
    Lrows = 1/2.5
    Lcols = 1.7
    theta = -27.5

    resampledImage = bicubicTransform(Lrows,Lcols,theta, image)

    # Second transform: Inverse of first transform
    # This transform is broken down into two separate transforms: one that resizes
    # the image back to its original size, and another that rotates it back to its original state.
    # The reason why this is necessary is because the padding that is added to the image in
    # the first transform makes the image ofcentered in the inverse transform.
    # The extra padding is removed after the image is resized so it can be rotated from
    # its center properly

    # First subtransform: bring the image back to original size
    Lrows = 2.5
    Lcols = 1/1.7
    theta = 0

    ReresampledImage = bicubicTransform(Lrows,Lcols,theta, resampledImage)

    # Compute proper dimension for the image and its center coordinates
    height = (round(abs(image.shape[0]*math.cos(theta))) + round(abs(image.shape[1]*math.sin(theta))))
    width = (round(abs(image.shape[1]*math.cos(theta))) + round(abs(image.shape[0]*math.sin(theta))))
    x0, y0 = (ReresampledImage.shape[0]//2, ReresampledImage.shape[1]//2)

    # crop image from its center into the correct size
    ReresampledImage_cropped = ReresampledImage[x0-height//2:x0+height//2, y0-width//2:y0+width//2]

    # second subtransform: Rotate back the image
    Lrows = 1
    Lcols = 1
    theta = 27.5

    finalImage = bicubicTransform(Lrows,Lcols,theta, ReresampledImage)

    # crop final image to original image size
    finalImage_cropped = finalImage[finalImage.shape[0]//2-512//2:finalImage.shape[0]//2+512//2, finalImage.shape[1]//2-512//2:finalImage.shape[1]//2+512//2]

    # take difference between final and original image
    diffImage = abs(image - finalImage_cropped)

    # compute and print PSNR value of final image
    PSNRvalue = dip.PSNR(image,finalImage_cropped)
    print('PSNR: ', PSNRvalue)

    # save all image generated throughout the process
    cv.imwrite('resampledImage.png',resampledImage)
    cv.imwrite('ReresampledImage.png',ReresampledImage)
    cv.imwrite('finalImage.png',finalImage)
    cv.imwrite('finalImage_cropped.png',finalImage_cropped)
    cv.imwrite('diffImage.png',diffImage)

if __name__ == "__main__":
    main()