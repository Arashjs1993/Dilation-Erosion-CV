import numpy as np

#Erosion function
def erode_this(origImg, erosionLevel=3):
    erosionLevel = 3 if erosionLevel < 3 else erosionLevel

    structuringKernel = np.full(shape=(erosionLevel, erosionLevel), fill_value=255)

    origImgShape = origImg.shape
    paddingWidth = erosionLevel - 2

    # pad the matrix with `paddingWidth`
    paddedImg = np.pad(array=origImg, pad_width=paddingWidth, mode='constant')
    paddedImgShape = paddedImg.shape
    h_reduce, w_reduce = (paddedImgShape[0] - origImgShape[0]), (paddedImgShape[1] - origImgShape[1])

    # sub matrices of kernel size
    flat_submatrices = np.array([
        paddedImg[i:(i + erosionLevel), j:(j + erosionLevel)]
        for i in range(paddedImgShape[0] - h_reduce) for j in range(paddedImgShape[1] - w_reduce)
    ])

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    erodedImg = np.array([255 if (i == structuringKernel).all() else 0 for i in flat_submatrices])
    erodedImg = erodedImg.reshape(origImgShape)

    return erodedImg