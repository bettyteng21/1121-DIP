import numpy as np
import cv2
from matplotlib import pyplot as plt


def ideal_hp_filtering(origin_img):
    h, w = origin_img.shape[0], origin_img.shape[1]

    # 取得原圖的dft
    imgFloat32 = np.float32(origin_img)
    F = np.fft.fft2(imgFloat32)
    Fshift = np.fft.fftshift(F)

    # ideal high pass filter
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt(np.power(x - (w//2), 2)+np.power(y - (h//2), 2))
    hpf = np.zeros((h, w), np.uint8)
    d0 = 20
    shift = 1  # 最後整個HPF往上移多少

    for i in range(h):
        for j in range(w):
            if (D[i, j] > d0):
                hpf[i, j] = 1+shift
            else:
                hpf[i, j] = 0+shift

    # 相乘後再做idft回到spatial domain
    output_img = np.abs(np.fft.ifft2(Fshift * hpf))

    output_img = np.clip(output_img, 0, 255)

    return output_img


def gaussian_hp_filtering(origin_img):
    h, w = origin_img.shape[0], origin_img.shape[1]

    # 取得原圖的dft
    imgFloat32 = np.float32(origin_img)
    F = np.fft.fft2(imgFloat32)
    Fshift = np.fft.fftshift(F)

    # gaussian high pass filter
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt(np.power(x - (w//2), 2)+np.power(y - (h//2), 2))
    d0 = 20
    shift = 1  # 最後整個HPF往上移多少
    hpf = np.exp(((-1)*np.power(D, 2))/(2*np.power(d0, 2)))
    hpf = (1-hpf) + shift  # hpf = 1-lpf

    # 相乘後再做idft回到spatial domain
    output_img = np.abs(np.fft.ifft2(Fshift * hpf))
    output_img = np.clip(output_img, 0, 255)

    return output_img


def butterworth_hp_filtering(origin_img):
    h, w = origin_img.shape[0], origin_img.shape[1]

    # 取得原圖的dft
    imgFloat32 = np.float32(origin_img)
    F = np.fft.fft2(imgFloat32)
    Fshift = np.fft.fftshift(F)

    # butterworth high pass filter
    n = 2
    d0 = 20
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt(np.power(x - (w//2), 2)+np.power(y - (h//2), 2))
    epsilon = 1e-8  # 防止被 0 除
    shift = 1  # 最後整個HPF往上移多少
    hpf = 1.0 / (1.0 + (np.power((d0 / (D+epsilon)), (2 * n)))) + shift

    # 相乘後再做idft回到spatial domain
    output_img = np.abs(np.fft.ifft2(Fshift * hpf))
    output_img = np.clip(output_img, 0, 255)

    return output_img


def homomorphic_filtering(origin_img):

    height, width = origin_img.shape[0], origin_img.shape[1]

    # 對原圖取log，(1e-5)是為了避免log0發生
    origin_img = np.float32(np.log(1e-5 + origin_img))

    # 取得原圖的dft
    F = np.fft.fft2(origin_img)

    # homomorphc filter
    rl, rh = 0.5, 2
    d0 = 50
    c = 1
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    D = np.sqrt(np.power(x - (width//2), 2)+np.power(y - (height//2), 2))
    H = (rh-rl)*(1-np.exp(((-1)*c*(np.power(D, 2)))/np.power(d0, 2))) + rl

    # 相乘後再做idft回到spatial domain
    dst_ifft = np.fft.ifft2(H*F).real

    # 因為最一開始取了log，這裡取exponential轉回來
    output_img = np.exp(dst_ifft)-(1e-5)

    output_img = cv2.normalize(
        output_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    for i in range(output_img.shape[0]):
        for j in range(output_img.shape[1]):
            if (output_img[i, j] < 45):
                output_img[i, j] = 0

    return np.float32(output_img)


def freq_laplacian(origin_img):

    h, w = origin_img.shape[0], origin_img.shape[1]

    # 把圖片變成0~1區間
    origin_scaled = np.float32(origin_img / 255)

    # 取得原圖的dft
    F = np.fft.fft2(origin_scaled)

    # frequency domain laplacian
    H = np.zeros((h, w), dtype=np.float32)
    for u in range(h):
        for v in range(w):
            H[u, v] = -4*np.power(np.pi, 2)*((u-(h/2))**2 + (v-(w/2))**2)

    # 相乘後再做idft回到spatial domain
    laplacian = (np.fft.ifft2(H*F)).real

    laplacian_scaled = cv2.normalize(
        laplacian, None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 𝑔(𝑥, 𝑦) = 𝑓(𝑥, 𝑦) + 𝑐(∇𝑓(𝑥, 𝑦))^2
    c = -0.45
    output_img = origin_scaled + (c*laplacian_scaled)

    output_img = np.clip(output_img, 0, 1)

    output_img = cv2.normalize(
        output_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return np.float32(output_img)


def highboost_filtering(origin_img):

    n = 2
    d0 = 20
    h, w = origin_img.shape[0], origin_img.shape[1]

    # 取得原圖的dft
    origin_img = np.float32(origin_img)
    F = np.fft.fft2(origin_img)
    Fshift = np.fft.fftshift(F)

    # butterworth Low Pass Filter
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt(np.power(x - (w//2), 2)+np.power(y - (h//2), 2))
    epsilon = 1e-8  # 防止被 0 除
    lpf = 1.0 / (1.0 + (np.power((D / (d0+epsilon)), (2 * n))))

    # 課本eq 4-131
    k = 3
    G = (1 + k*(1-lpf))*Fshift
    output_img = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))

    output_img = np.clip(output_img, 0, 255)

    return np.float32(output_img)


def plot_two_histo(origin_img, processed_img, process_title):

    file_name = "./images/comparisons/4_comparison_"+process_title+".jpg"

    plt.subplot(2, 2, 1)
    plt.title('origin_img')
    plt.axis('off')
    plt.imshow(origin_img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title(process_title)
    plt.axis('off')
    plt.imshow(processed_img, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('origin_img')
    histr_origin = cv2.calcHist([origin_img], [0], None, [256], [0, 256])
    plt.plot(histr_origin)

    plt.subplot(2, 2, 4)
    plt.title(process_title)
    histr_processed = cv2.calcHist([processed_img], [0], None, [256], [0, 256])
    plt.plot(histr_processed)
    plt.savefig(file_name, bbox_inches='tight', dpi=500)
    plt.show()

    return


if __name__ == "__main__":
    origin_img = cv2.imread(
        "./images/angiogram_aortic_kidney.tif", cv2.IMREAD_GRAYSCALE)
    origin_img = np.asarray(origin_img)

    plt.subplot(2, 2, 1)
    plt.title('Original img')
    plt.imshow(origin_img, cmap='gray')
    plt.axis('off')

    ideal_img = ideal_hp_filtering(origin_img)
    plt.subplot(2, 2, 2)
    plt.title('ideal HPF')
    plt.imshow(ideal_img, cmap='gray')
    plt.axis('off')

    gaussian_img = gaussian_hp_filtering(origin_img)
    plt.subplot(2, 2, 3)
    plt.title('gaussian HPF')
    plt.imshow(gaussian_img, cmap='gray')
    plt.axis('off')

    butter_img = butterworth_hp_filtering(origin_img)
    plt.subplot(2, 2, 4)
    plt.title('butterworth HPF')
    plt.imshow(butter_img, cmap='gray')
    plt.axis('off')
    plt.savefig('./images/comparisons/4_comparison_HPFs.jpg',
                bbox_inches='tight', dpi=500)
    plt.show()

    laplacian_img = freq_laplacian(origin_img)
    plot_two_histo(np.float32(origin_img), laplacian_img,
                   "freq_laplacian_filtering")

    homomorphic_img = homomorphic_filtering(origin_img)
    plot_two_histo(np.float32(origin_img), homomorphic_img,
                   "homomorphic_filtering")

    highboost_img = highboost_filtering(origin_img)
    plot_two_histo(np.float32(origin_img), highboost_img,
                   "highboost_filtering")

    cv2.imwrite('./images/4_ideal_HPF_filtering.jpg', ideal_img)
    cv2.imwrite('./images/4_gaussian_HPF_filtering.jpg', gaussian_img)
    cv2.imwrite('./images/4_butterworth_HPF_filtering.jpg', butter_img)
    cv2.imwrite('./images/4_freq_laplacian_filtering.jpg', laplacian_img)
    cv2.imwrite('./images/4_homomorphic_filtering.jpg', homomorphic_img)
    cv2.imwrite('./images/4_highboost_filtering.jpg', highboost_img)
