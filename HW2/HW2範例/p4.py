import numpy as np
import cv2

# Shift mask
def shift_mask(H, W):
    Hmap, Wmap = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')
    mask = np.power(np.zeros((H, W)) - 1, Hmap + Wmap)

    return mask

# Ideal lowpass filter
def ILPF(padded_img, D_0):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2)
    h = np.zeros(D_square.shape)
    h[D_square <= np.power(D_0, 2)] = 1
    
    return h

# Butterworth lowpass filter
def BLPF(padded_img, D_0, n):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2)
    h = 1 / (1 + (np.power(D_square, n) / np.power(D_0, (2 * n))))
    
    return h

# Ideal highpass filter
def IHPF(padded_img, D_0=50):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2)
    h = np.zeros(D_square.shape)
    h[D_square > np.power(D_0, 2)] = 1
    
    return h

# Laplacian
def laplacian(padded_img):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2)
    h = -4 * np.power(np.pi, 2) * D_square

    return h

# Unsharp masking and highboost filtering
def unsharp_masking(padded_img, k1, k2):
    h_hp = IHPF(padded_img, D_0=50)
    h = k1 + k2 * h_hp

    return h

# Gaussian band reject filter
def GBRF(padded_img, Co, W):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2) + 1e-5
    h = 1 - np.exp(-np.power((D_square - np.power(Co, 2) / np.power(D_square, 0.5) * W), 2))

    return h

# Homomorphic filtering
def homomorphic(padded_img, gamma_h, gamma_l, c, D_0):
    P, Q = padded_img.shape
    Hmap, Wmap = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')
    D_square = np.power((Hmap - P // 2), 2) + np.power((Wmap - Q // 2), 2) + 1e-5
    h = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_square / np.power(D_0, 2)))) + gamma_l

    return h

# Filtering in frequency domain, steps referred to Textbook
def filtering(img, func="ILPF"):
    H, W = img.shape
    padded_img = (img * shift_mask(H, W))
    # Pad to 2M x 2N
    padded_img = np.pad(padded_img, ((0, H), (0, W))).astype(np.float64)
    P, Q = padded_img.shape
    match func:
        case "ILPF":
            f = np.fft.fft2(padded_img)
            h = ILPF(padded_img, D_0=300)
        case "BLPF":
            f = np.fft.fft2(padded_img)
            h = BLPF(padded_img, D_0=200, n=1.5)
        case "IHPF":
            f = np.fft.fft2(padded_img)
            h = IHPF(padded_img, D_0=70)
        case "laplacian":
            f = np.fft.fft2(padded_img / 255)
            h = laplacian(padded_img)
        case "unsharp":
            f = np.fft.fft2(padded_img)
            h = unsharp_masking(padded_img, k1=50, k2=5)
        case "GBRF":
            f = np.fft.fft2(padded_img)
            h = GBRF(padded_img, Co=160, W=120)
        case "homomorphic":
            padded_img = np.pad(img, ((0, H), (0, W))).astype(np.uint8)
            # To avoid the zero division
            padded_img = np.log(padded_img + 1e-3)
            f = np.fft.fft2(padded_img)
            h = homomorphic(padded_img, 2.5, 0.2, 1, 70)
    filtered_img = (np.real(np.fft.ifft2(f * h)) * shift_mask(P, Q))[:H, :W]
    if func == "laplacian":
        # For laplacian, normalization is different 
        # Based on textbook
        filtered_img /= np.max(filtered_img)
        filtered_img = img - filtered_img
    elif func == "homomorphic":
        # Different for homomorphic
        filtered_img = (np.real(np.fft.ifft2(f * h)))[:H, :W]
        filtered_img = np.exp(filtered_img) - 1e-3
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
    
    return filtered_img.astype(np.uint8)

def enhance(img):
    enhanced_img = filtering(img, "ILPF")
    enhanced_img = filtering(enhanced_img, "laplacian")
    enhanced_img = filtering(enhanced_img, "unsharp")
    enhanced_img = filtering(enhanced_img, "GBRF")
    enhanced_img = filtering(img, "homomorphic")

    return enhanced_img

if __name__ == "__main__":
    img = cv2.imread("./images/angiogram_aortic_kidney.tif", cv2.IMREAD_GRAYSCALE)
    enhanced_img = enhance(img)
    cv2.imshow("Original image", img)
    cv2.imshow("Enhanced image", enhanced_img)
    # cv2.imwrite("enhanced_img_p4.png", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 