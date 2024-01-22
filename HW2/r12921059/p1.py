import numpy as np
import cv2
from matplotlib import pyplot as plt


def gamma_transformation(origin_img, gamma):
    origin_img = origin_img.astype(np.uint8)

    # 把對應的值畫成table，之後直接查表
    gamma_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    output_img = cv2.LUT(origin_img, gamma_table)

    return output_img


def histogram_equalize(origin_img):
    return cv2.equalizeHist(origin_img)


def sobel(origin_img):
    '''
    xorder = 0, yorder = 1, ksize = 3

    Sobel kernel:
    [[-1,  0,  1],
     [-2,  0,  2],
     [-1,  0,  1]]


    xorder = 1, yorder = 0, ksize = 3

    Sobel kernel:
    [[-1  -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]]
    '''

    grad_x = cv2.Sobel(origin_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(origin_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # vertical和horizontal sobel的權重各為0.5，然後相加
    output_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return output_img


def unsharp_mask(origin_img, k):
    median_img = cv2.medianBlur(origin_img, 13)

    # output = (1+k)*origin - k*blurred
    output_img = cv2.addWeighted(origin_img, 1+k, median_img, -k, 0)

    return output_img


def method1(origin_img):
    # 取得sobel image
    sobel_img = sobel(origin_img)
    sobel_median_img = np.float32(cv2.medianBlur(sobel_img, 13))
    sobel_median_img = cv2.normalize(sobel_median_img, None, 0, 255,
                                     cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 取得laplacian image
    laplacian_img = cv2.Laplacian(origin_img, cv2.CV_32F, ksize=3)
    laplacian_img = cv2.normalize(
        laplacian_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    enhance = sobel_median_img*laplacian_img
    enhance = cv2.normalize(enhance, None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # output = origin + k*enhance，這裡取k=0.8
    output_img = cv2.addWeighted(np.float32(origin_img), 1, enhance, 0.8, 0)

    for i in range(output_img.shape[0]):
        for j in range(output_img.shape[1]):
            if(output_img[i, j] < 118):
                output_img[i, j] = 0
            if(output_img[i, j] > 255):
                output_img[i, j] = 255

    return output_img


def method2(origin_img, r1, s1, r2, s2):
    # divided linear trandformation
    # 設(r1,s1)和(r2,s2)兩點，把img intensity切成三段，每段可以自訂要stretch/compress
    dlt = np.zeros(256)
    for i in range(len(dlt)):
        if (i < r1):
            # 計算(0,0)到(r1,s1)這條線所轉換的值
            dlt[i] = (s1/r1)*i
        elif (i < s2):
            # 計算(r1,s1)到(r2,s2)這條線所轉換的值
            dlt[i] = (s2-s1)/(r2-r1) * (i-r1) + s1
        else:
            # 計算(r2,s2)到(255,255)這條線所轉換的值
            dlt[i] = (255.0-s2)/(255.0-r2)*(i-r2) + s2

    # 查表
    stretched_img = cv2.LUT(origin_img, dlt)

    return np.uint8(stretched_img)


def plot_two_histo(origin_img, processed_img, process_title):
    file_name = "./images/comparisons/1_comparison_"+process_title+".jpg"

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
    origin_img = cv2.imread("./images/angiogram_aortic_kidney.tif", 0)

    gamma_trans_img = gamma_transformation(origin_img, 2.2)
    plot_two_histo(origin_img, gamma_trans_img, "gamma_trans")

    histo_equal_img = histogram_equalize(origin_img)
    plot_two_histo(origin_img, histo_equal_img, "histo_equal")

    unsharp_mask_img = unsharp_mask(origin_img, 1)
    plot_two_histo(origin_img, unsharp_mask_img, "unsharp_mask")

    method1_img = method1(origin_img)
    plot_two_histo(origin_img, method1_img, "method1")

    method2_img = method2(origin_img, 100, 10, 130, 70)
    plot_two_histo(origin_img, method2_img, "method2")

    cv2.imwrite('./images/1_gamma_transformation.jpg', gamma_trans_img)
    cv2.imwrite('./images/1_histogram_equalization.jpg', histo_equal_img)
    cv2.imwrite('./images/1_unsharp_masking.jpg', unsharp_mask_img)
    cv2.imwrite('./images/1_method1.jpg', method1_img)
    cv2.imwrite('./images/1_method2.jpg', method2_img)
