import numpy as np

#Function for dilation
def dilate_this(origImg, dilationLevel=3):
    # setting the dilationLevel
    dilationLevel = 3 if dilationLevel < 3 else dilationLevel
    
    # obtain the kernel by the shape of (dilationLevel, dilationLevel)
    structuringKernel = np.full(shape=(dilationLevel, dilationLevel), fill_value=255)
    
    origImgShape = origImg.shape
    print("Original image shape: ", origImgShape)
    paddingWidth = dilationLevel - 2
    
    # pad the image with paddingWidth
    paddedImg = np.pad(array=origImg, pad_width=paddingWidth, mode='constant')
    padImgShape = paddedImg.shape
    print("padded image shape: ", padImgShape)
    h_reduce, w_reduce = (padImgShape[0] - origImgShape[0]), (padImgShape[1] - origImgShape[1])
    
    # obtain the submatrices according to the size of the kernel
    flat_submatrices = np.array([
        paddedImg[i:(i + dilationLevel), j:(j + dilationLevel)]
        for i in range(padImgShape[0] - h_reduce) for j in range(padImgShape[1] - w_reduce)
    ])
    
    # replace the values either 255 or 0 by dilation condition
    dilatedImg = np.array([255 if (i == structuringKernel).any() else 0 for i in flat_submatrices])
    # obtain new matrix whose shape is equal to the original image size
    dilatedImg = dilatedImg.reshape(origImgShape)
    
    return dilatedImg