# Import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = np.float32(cv2.imread(path + "source_" + id + ".jpg", 1))
    target = np.float32(cv2.imread(path + "target_" + id + ".jpg", 1))
    mask   = np.float32(cv2.imread(path + "mask_" + id + ".jpg", 1))
    return source, mask, target


# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Pyramid Blend
def PyramidBlend(source, mask, target, numLayers):    
    # Generate Gaussians for source, mask and target
    # Temp images that are scaled down every iteration
    GS = source.copy()
    GT = target.copy()
    GM = mask.copy()
    # The pyramids
    gSource = [GS]
    gTarget = [GT]
    gMask = [GM]
    for i in range(0, numLayers):
        # scale down
        GS = cv2.pyrDown(GS)
        GT = cv2.pyrDown(GT)
        GM = cv2.pyrDown(GM)
        # append to pyramid
        gSource.append(GS)
        gTarget.append(GT)
        gMask.append(GM)

    # Generate Laplacian for source and target
    lpSource = [gSource[numLayers - 1]]
    lpTarget = [gTarget[numLayers - 1]]
    # Flip the gMask so the layers match the laplacians
    gMaskFlipped = [gMask[numLayers - 1]]
    for i in range(numLayers - 1, 0, -1):
        # get high frequencies by subtracting
        size = (gSource[i-1].shape[1], gSource[i-1].shape[0])
        LS = np.subtract(gSource[i-1], cv2.pyrUp(gSource[i], dstsize=size))
        size = (gTarget[i-1].shape[1], gTarget[i-1].shape[0])
        LT = np.subtract(gTarget[i-1], cv2.pyrUp(gTarget[i], dstsize=size))
        
        # Add to the pyramid
        lpSource.append(LS)
        lpTarget.append(LT)

        gMaskFlipped.append(gMask[i - 1])
    
    # Perform the Calculation 
    LCs = []
    for Ls, Lt, Gm in zip(lpSource, lpTarget, gMaskFlipped):
        # Given formula
        lc = Ls * Gm + Lt * (1.0 - Gm)
        LCs.append(lc)

    # Reconstruct Image at each level
    finalImage = LCs[0]
    for i in range(1, numLayers):
        size = (LCs[i].shape[1], LCs[i].shape[0])
        # Scale it up and add to the final image
        finalImage = cv2.pyrUp(finalImage, dstsize=size)
        finalImage = cv2.add(finalImage, LCs[i])

    print("Pyramid Done!")
    return finalImage

def createMatrixA(numRows, numCols):
    # use scipy.sparse
    D = scipy.sparse.lil_matrix((numRows, numRows))
    D.setdiag(-1, -1)
    D.setdiag(4)
    D.setdiag(-1, 1)
    A = scipy.sparse.block_diag([D] * numCols).tolil()
    A.setdiag(-1, 1 * numRows)
    A.setdiag(-1, -1 * numRows)
    
    return A

# Poisson Blend
def PoissonBlend(source, mask, target, isMix):
    
    # Get the size of the target
    maxY, maxX = target.shape[:-1]
    A = createMatrixA(maxX, maxY)

    converted = A.tocsc()

    for y in range(1, maxY - 1):
        for x in range(1, maxX - 1):
            if mask[y][x] == 0: # region outside of mask
                # Update A matrix
                k = x + y * maxX
                A[k, k] = 1
                A[k, k + 1] = 0
                A[k, k - 1] = 0
                A[k, k + maxX] = 0
                A[k, k - maxX] = 0

    A = A.tocsc()
    flatMask = mask.flatten()
        
    # Go through each color channel
    for channel in range(source.shape[2]):
        # Flatten matrices for given channel
        flatSource = source[0:maxY, 0:maxX, channel].flatten()
        flatTarget = target[0:maxY, 0:maxX, channel].flatten()        

        # create B matrix
        B = converted.dot(flatSource)
        B[flatMask==0] = flatTarget[flatMask==0]
        
        # Solve the matrix
        x = spsolve(A, B)

        # Normalize the solution
        x = x.reshape((maxY, maxX))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        # Set target's channel to solution x
        target[0:maxY, 0:maxX, channel] = x

    print("Poisson Done!")

    return target

def NaiveBlend(source, target, mask):
    return source * mask + target * (1 - mask)
    
if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'
    
    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[0, 0], [0, 0], [210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88], [350, 200]]

    # main area to specify files and display blended image
    for index in range(1, len(offsets)):

        print ("Doing " + str(index))
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 127] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        ### The main part of the code ###
    
        # Implement the PyramidBlend function (Task 1)
        pyramidOutput = PyramidBlend(source, mask, target, 6)
        cv2.imwrite("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)

        maskGrayscale = []
        for i in range(0, mask.shape[0]):
            temp = []
            for j in range(0, mask.shape[1]):
                if np.any(mask[i][j]):
                    temp.append(1)
                else:
                    temp.append(0)

            maskGrayscale.append(temp)

        maskGrayscale = np.array(maskGrayscale)
        # Implement the PoissonBlend function (Task 2)
        poissonOutput = PoissonBlend(source, maskGrayscale, target, isMix)
        
        # Writing the result

        if not isMix:
            cv2.imwrite("{}poisson_{}.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
        else:
            cv2.imwrite("{}poisson_{}_Mixing.jpg".format(outputDir, str(index).zfill(2)), poissonOutput)
