import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function for basic convolution
def convolve(img, kernel, padding, n):
    H, W = img.shape
    conv_img = np.zeros((H, W))
    ref = np.pad(img, padding, 'constant')
    for row in range(H):
        for col in range(W):
            conv_img[row][col] = np.sum(kernel * ref[row:row + n, col:col + n])
    
    return conv_img

# Gamma transformation
def gamma(img, gamma=1, c=1, epsilon=0):
    gamma_img = c * np.power((img / 255) + epsilon, gamma) * 255
    
    return gamma_img.astype(np.uint8)

# Histogram equalization
def hist_eq(img, mode='global', gray_level=256):
    H, W = img.shape
    hist_img = np.zeros(img.shape)
    hist = np.zeros(gray_level, dtype=int)
    mapping = np.zeros(gray_level)
    for row in range(H):
        for col in range(W):
            hist[img[row, col]] += 1
    hist = hist / (H * W)
    # plt.bar(np.arange(0, gray_level), hist)
    mapping[0] = hist[0]
    for x in range(1, gray_level):
        mapping[x] = mapping[x - 1] + hist[x]
    mapping = np.round((gray_level - 1) * mapping)
    for row in range(H):
        for col in range(W):
            hist_img[row, col] = mapping[img[row, col]]

    return hist_img.astype(np.uint8)

# Box filter
def box_filter(img, padding, n):
    # n * n kernel
    kernel = np.empty(shape=(n, n))
    kernel.fill(1 / (n * n))
    blurred_img = convolve(img, kernel, padding, n)

    return blurred_img.astype(np.uint8)

# Unsharp masking (using generated blurred image)
def unsharp_masking(img, blurred_img, k=1):
    mask = img - blurred_img
    highboost_img = img + k * mask

    return highboost_img.astype(np.uint8)

# Median filter
def median(img, padding, n=3):
    H, W  = img.shape
    median_img = np.zeros((H, W))
    ref = np.pad(img, padding, 'constant')
    for row in range(H):
        for col in range(W):
            median_img[row, col] = np.median(ref[row : row + n, col : col + n].flatten())

    return median_img.astype(np.uint8)

# Laplacian 
def laplacian(img):
    kernel = np.array([[0, 1, 0], 
                      [1, -4, 1], 
                      [0, 1, 0]])
    laplacian_img = convolve(img, kernel, ((1, 1), (1, 1)), 3)

    return laplacian_img.astype(np.uint8)

# Sobel
def sobel(img):
    kernel1 = np.array([[-1, -2, -1], 
                      [0, 0, 0], 
                      [1, 2, 1]])
    kernel2 = np.array([[-1, 0, 1], 
                      [-2, 0, 2], 
                      [-1, 0, 1]])
    gx = convolve(img, kernel1, ((1, 1), (1, 1)), 3)
    gy = convolve(img, kernel2, ((1, 1), (1, 1)), 3)
    enhanced_img = np.power(np.power(gx, 2) + np.power(gy, 2), 0.5)

    return enhanced_img.astype(np.uint8)

def enhance(img):
    enhanced_img = box_filter(img, ((1, 1), (1, 1)), n=3)
    enhanced_img = median(enhanced_img, ((1, 1), (1, 1)), n=3)
    enhanced_img = unsharp_masking(enhanced_img, enhanced_img, k=2.85) 
    enhanced_img = hist_eq(enhanced_img)
    enhanced_img = gamma(enhanced_img, gamma=1.95, c=1, epsilon=0)

    return enhanced_img

if __name__ == "__main__":
    img = cv2.imread("./images/angiogram_aortic_kidney.tif", cv2.IMREAD_GRAYSCALE)
    enhanced_img = enhance(img)
    cv2.imshow("Original image", img)
    cv2.imshow("Enhanced image", enhanced_img)
    # cv2.imwrite("enhanced_img_p1.png", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 