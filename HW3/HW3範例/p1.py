import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy

image_path = "images/motion blur_1_3024x4032.jpg"
image = cv2.imread(image_path, 0)
image = image.astype(float)
image = image / 255
#cv2.imwrite('origin_keyboard.png', image)

donald_duck_path = "images/Donald_Duck.jpeg"
donald_duck = cv2.imread(donald_duck_path, 0)
donald_duck = donald_duck.astype(float)
donald_duck = donald_duck / 255
#cv2.imwrite('origin_donald_duck.png', donald_duck)

# Inverse fourier transform
def ifft(img):
    enhanced_image = np.fft.ifftshift(img)
    enhanced_image = np.abs(np.fft.ifft2(enhanced_image)) * 255
    return enhanced_image

# Fourier transform
def fft(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    print(f_transform_shifted.shape)
    return f_transform_shifted

def wiener_filter(h, f, k):
    mag = np.conj(h) * h
    r,c = h.shape[0], h.shape[1]
    over_h = np.zeros((r, c), dtype=np.complex128)
    for u in range(r):
        for v in range(c):
            if h[u][v] != 0:
                over_h[u][v] = 1 / h[u][v]
    kernel = over_h * (mag / (mag + k))
    dummy = f * kernel
    return dummy

def H(image, a, b, T):
    r,c = image.shape[0], image.shape[1]
    h = np.zeros((r, c), dtype=np.complex128)
    for u in range(r):
        for v in range(c):
            if (u * a + v * b) != 0:
                h[u][v] = (T / (np.pi * (u * a + v * b))) * np.sin(np.pi * (u * a + v * b)) * np.exp(-1j * (np.pi * (u * a + v * b)))
    return h

#code for keyboard
f_image = fft(image)
hf = H(f_image, 0.002, -0.002, 1)
f_dummy = wiener_filter(hf, f_image, 0.1)
new_f = ifft(f_dummy)
new_f = cv2.subtract(image * 255 , new_f)
cv2.imwrite('restored_keyboard.png', new_f)

#code for donald duck
d_image = fft(donald_duck)
hd = H(d_image, 0, 0.1, 30)
d_dummy = wiener_filter(hd, d_image, 0.1)
new_d = ifft(d_dummy)
new_d = cv2.subtract(don0ald_duck * 255 , new_d)

cv2.imwrite('restored_donald_duck.png', new_d)