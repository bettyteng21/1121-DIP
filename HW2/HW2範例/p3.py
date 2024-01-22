import numpy as np
import cv2

# Shift mask (-1)^(x+y)
def shift_mask(H, W):
    Hmap, Wmap = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')
    mask = np.power(np.zeros((H, W)) - 1, Hmap + Wmap)

    return mask

# Function for basic convolution
def convolve(img, kernel, padding, n):
    H, W = img.shape
    conv_img = np.zeros((H, W))
    ref = np.pad(img, padding, 'constant')
    for row in range(H):
        for col in range(W):
            conv_img[row][col] = np.sum(kernel * ref[row:row + n, col:col + n])
    
    return conv_img

# Sobel filtering using vertical kernel in frequency domain
# According to example
def f_sobel_filter(img, sobel_kernel):
    H, W = img.shape
    P, Q = H + 3 - 1, W + 3 - 1
    padded_img = np.zeros((P, Q))
    padded_img[:H, :W] = img * shift_mask(H, W)
    f = np.fft.fft2(padded_img)
    padded_kernel = np.zeros((P, Q))
    H_k, W_k = sobel_kernel.shape
    padded_kernel[:H_k, :W_k] = sobel_kernel * shift_mask(H_k, W_k)
    h = np.fft.fft2(padded_kernel)
    filtered_img = (np.abs(np.fft.ifft2(f * h)) * shift_mask(P, Q))[:H, :W]
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)

    return filtered_img.astype(np.uint8)

# Sobel filtering using vertical kernel in spatial domain
def s_sobel_filter(img, sobel_op):
    filtered_img = convolve(img, sobel_op, padding=((1, 1), (1, 1)), n=3)
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)

    return filtered_img.astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_db = (20 * np.log10(np.abs(f_shift))) / 255
    # cv2.imwrite("f_spectrum_keyboard.png", f_db * 255)
    sobel_op = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_kernel = np.zeros((4, 4))
    sobel_kernel[1:, 1:] = sobel_op
    f_filtered_img = f_sobel_filter(img, sobel_kernel)
    s_filtered_img = s_sobel_filter(img, sobel_op)
    f_even_filtered_img = f_sobel_filter(img, sobel_op)
    print(np.equal(f_filtered_img, f_even_filtered_img))
    cv2.imshow("f_filtered_keyboard.png", f_filtered_img)
    # cv2.imwrite("f_filtered_keyboard.png", f_filtered_img)
    cv2.imshow("s_filtered_keyboard.png", s_filtered_img)
    # cv2.imwrite("s_filtered_keyboard.png", s_filtered_img)
    cv2.imshow("f_without_enforce_filtered_keyboard.png", f_even_filtered_img)
    # cv2.imwrite("f_without_enforce_filtered_keyboard.png", f_even_filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 