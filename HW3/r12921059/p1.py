import numpy as np
import cv2
from matplotlib import pyplot as plt


def blur_img(origin_img, weight):
    blur_img = cv2.GaussianBlur(origin_img, (5, 5), 0)

    return (weight*origin_img)+((1-weight)*blur_img)


def pad_kernel(image, rows, cols):
    image_pad = np.zeros((rows, cols))

    # 把kernel pad到圖片中央
    mid_row = rows // 2
    mid_col = cols // 2
    image_pad[(mid_row-1):(mid_row+2), (mid_col-1):(mid_col+2)] = image

    return image_pad


def gamma_transformation(origin_img, gamma):
    origin_img = origin_img.astype(np.uint8)

    # 把對應的值畫成table，之後直接查表
    gamma_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    output_img = cv2.LUT(origin_img, gamma_table)

    return output_img


def brightness_contrast(img, brightness, contrast):
    # 調整圖片的brightness和contrast
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def new_filtering(origin_img):
    # Improve image de-blurring，參考致這篇paper
    # https://ieeexplore.ieee.org/document/8316595

    markov_basis = np.array([[1, -1,  0],
                             [-1,  0,  0],
                             [0,  0,  0]])
    laplacian_filter = np.array([[-1, -2, -1],
                                 [-2, 12, -2],
                                 [-1, -2, -1]])

    new_filter = markov_basis+laplacian_filter

    # for better performance in colored-image
    new_filter[1, 1] = new_filter[1, 1]+1

    output_img = cv2.filter2D(origin_img, cv2.CV_32F,
                              new_filter, None, (-1, -1), 0, cv2.BORDER_CONSTANT)

    return output_img


def motion_function(h, w, angle, dist):
    x_center = (h-1)//2
    y_center = (w-1)//2

    sin_val = np.sin(angle*np.pi/180)
    cos_val = np.cos(angle*np.pi/180)
    PSF = np.zeros((h, w))

    for i in range(dist):
        x_offset = round(sin_val*i)
        y_offset = round(cos_val*i)

        PSF[(x_center-x_offset), (y_center-y_offset)] = 1

    normalized = PSF/PSF.sum()

    return normalized


def motion_kernel(angle, d, sz=65):

    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def idealLPFilter(img, radius=180):

    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    D = np.sqrt((u - M//2)**2 + (v - N//2)**2)
    kernel = np.zeros(img.shape[:2], np.float32)
    kernel[D <= radius] = 1
    return kernel


def butterworthLPFilter(img, n=2, d0=20):

    w, h = img.shape[1], img.shape[0]
    kernel = np.zeros((img.shape[0], img.shape[1]), np.float32)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt(np.power(x - (w//2), 2)+np.power(y - (h//2), 2))
    kernel = 1.0 / (1.0 + (np.power((D / d0), (2 * n))))
    return kernel


def inverse_filter(origin_img, PSF, epsilon=1e-8, padding=True, n=2, d0=20, b_gamma=1.0, g_gamma=1.0, r_gamma=1.0, brightness=50, contrast=100):

    origin_img, PSF = np.float32(origin_img), np.float32(PSF)
    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]

    if padding == True:
        # pad input image
        image_padding = len(PSF)//2
        image_pad = np.pad(origin_img, pad_width=[(
            image_padding, image_padding), (image_padding, image_padding), (0, 0)], mode='wrap')

        # pad PSF kernel
        pad_0_low = image_pad.shape[0] // 2 - PSF.shape[0] // 2
        pad_0_high = image_pad.shape[0] - PSF.shape[0] - pad_0_low
        pad_1_low = image_pad.shape[1] // 2 - PSF.shape[1] // 2
        pad_1_high = image_pad.shape[1] - PSF.shape[1] - pad_1_low
        PSF_pad = np.pad(PSF, ((pad_0_low, pad_0_high),
                               (pad_1_low, pad_1_high)), mode='constant')
    else:
        image_pad = origin_img
        PSF_pad = PSF

    # 初始化等一下會用到的東西
    F_inverse = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    F_shift = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    inverse_img = np.float32(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    output_img = np.float32(
        np.zeros((origin_h, origin_w, 3)))  # un-padded size

    F_PSF = np.fft.fftshift(np.fft.fft2(PSF_pad)) + epsilon

    # do computation on each BGR channel individually
    for i in range(3):
        F_shift[:, :, i] = np.fft.fftshift(np.fft.fft2(image_pad[:, :, i]))

        lpFilter = butterworthLPFilter(image_pad, n, d0)
        # lpFilter = idealLPFilter(image_pad, radius=180) # 也可用idealLPF，r=180是keyboard比較好的參數

        F_inverse[:, :, i] = (F_shift[:, :, i]/F_PSF) * lpFilter

        F_inverse[:, :, i] = np.fft.ifft2(np.fft.ifftshift(F_inverse[:, :, i]))

        inverse_img[:, :, i] = (np.fft.ifftshift(F_inverse[:, :, i])).real

        inverse_img[:, :, i] = inverse_img[:, :, i] + \
            new_filtering(inverse_img[:, :, i])

        inverse_img[:, :, i] = cv2.normalize(
            inverse_img[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if padding == True:
            # crop padded image back to original size
            output_img[:, :, i] = inverse_img[image_padding:-
                                              image_padding, image_padding:-image_padding, i]
        else:
            output_img[:, :, i] = inverse_img[:, :, i]

    # do some color modification to make the output look more close to original input
    output_img[:, :, 0] = gamma_transformation(output_img[:, :, 0], b_gamma)
    output_img[:, :, 1] = gamma_transformation(output_img[:, :, 1], g_gamma)
    output_img[:, :, 2] = gamma_transformation(output_img[:, :, 2], r_gamma)

    output_img = brightness_contrast(output_img, brightness, contrast)
    # output_img = brightness_contrast(output_img, brightness=150, contrast=200) # keyboard idealLPF的參數

    return output_img


def wiener_filter(origin_img, PSF, epsilon=1e-8, padding=True, K=0.01, b_gamma=1.0, g_gamma=1.0, r_gamma=1.0, brightness=50, contrast=100):

    origin_img, PSF = np.float32(origin_img), np.float32(PSF)
    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]

    if padding == True:
        # pad input image
        image_padding = len(PSF)//2
        image_pad = np.pad(origin_img, pad_width=[(
            image_padding, image_padding), (image_padding, image_padding), (0, 0)], mode='wrap')

        # pad PSF kernel
        pad_0_low = image_pad.shape[0] // 2 - PSF.shape[0] // 2
        pad_0_high = image_pad.shape[0] - PSF.shape[0] - pad_0_low
        pad_1_low = image_pad.shape[1] // 2 - PSF.shape[1] // 2
        pad_1_high = image_pad.shape[1] - PSF.shape[1] - pad_1_low
        PSF_pad = np.pad(PSF, ((pad_0_low, pad_0_high),
                               (pad_1_low, pad_1_high)), mode='constant')
    else:
        image_pad = origin_img
        PSF_pad = PSF

    # 初始化等一下會用到的東西
    F_wiener = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    F_origin = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    F_filtered = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    inverse_img = np.float32(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    output_img = np.float32(
        np.zeros((origin_h, origin_w, 3)))  # un-padded size

    F_PSF = (np.fft.fft2(PSF_pad))+epsilon

    # do computation on each BGR channel individually
    for i in range(3):
        F_origin[:, :, i] = (np.fft.fft2(image_pad[:, :, i]))

        F_wiener = np.conj(F_PSF)/((F_PSF*np.conj(F_PSF)) + K)

        F_filtered[:, :, i] = np.fft.ifft2(F_wiener*F_origin[:, :, i])

        inverse_img[:, :, i] = (np.fft.ifftshift(F_filtered[:, :, i])).real

        inverse_img[:, :, i] = inverse_img[:, :, i] + \
            new_filtering(inverse_img[:, :, i])

        inverse_img[:, :, i] = cv2.normalize(
            inverse_img[:, :, i], None, 0, 255, cv2.NORM_MINMAX)

        if padding == True:
            # crop padded image back to original size
            output_img[:, :, i] = inverse_img[image_padding:-
                                              image_padding, image_padding:-image_padding, i]
        else:
            output_img[:, :, i] = inverse_img[:, :, i]

    # do some color modification to make the output look more close to original input
    output_img[:, :, 0] = gamma_transformation(output_img[:, :, 0], b_gamma)
    output_img[:, :, 1] = gamma_transformation(output_img[:, :, 1], g_gamma)
    output_img[:, :, 2] = gamma_transformation(output_img[:, :, 2], r_gamma)

    output_img = brightness_contrast(output_img, brightness, contrast)

    return output_img


def CLS_filter(origin_img, PSF, epsilon=1e-8, padding=True, gamma=0.01, brightness=50, contrast=100):

    origin_img, PSF = np.float32(origin_img), np.float32(PSF)
    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]

    if padding == True:
        # pad input image
        image_padding = len(PSF)//2
        image_pad = np.pad(origin_img, pad_width=[(
            image_padding, image_padding), (image_padding, image_padding), (0, 0)], mode='wrap')

        # pad PSF kernel
        pad_0_low = image_pad.shape[0] // 2 - PSF.shape[0] // 2
        pad_0_high = image_pad.shape[0] - PSF.shape[0] - pad_0_low
        pad_1_low = image_pad.shape[1] // 2 - PSF.shape[1] // 2
        pad_1_high = image_pad.shape[1] - PSF.shape[1] - pad_1_low
        PSF_pad = np.pad(PSF, ((pad_0_low, pad_0_high),
                               (pad_1_low, pad_1_high)), mode='constant')
    else:
        image_pad = origin_img
        PSF_pad = PSF

    # 初始化等一下會用到的東西
    F_origin = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    F_filtered = np.complex128(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    inverse_img = np.float32(
        np.zeros((image_pad.shape[0], image_pad.shape[1], 3)))
    output_img = np.float32(
        np.zeros((origin_h, origin_w, 3)))  # un-padded size

    # laplacian kernel
    kernel = np.array([[0, -1,  0],
                       [-1,  4, -1],
                       [0, -1,  0]])

    # 取kernel們的fft
    F_PSF = np.fft.fft2(PSF_pad)+epsilon
    F_kernel = np.fft.fft2(pad_kernel(
        kernel, image_pad.shape[0], image_pad.shape[1]))

    F_CLS = F_PSF.conj() / ((F_PSF*np.conj(F_PSF))+(gamma*(F_kernel*np.conj(F_kernel))))

    # do computation on each BGR channel individually
    for i in range(3):
        F_origin[:, :, i] = np.fft.fft2(image_pad[:, :, i])

        F_filtered[:, :, i] = F_origin[:, :, i] * F_CLS

        F_filtered[:, :, i] = np.fft.ifft2(F_filtered[:, :, i])

        inverse_img[:, :, i] = (np.fft.ifftshift(F_filtered[:, :, i])).real

        inverse_img[:, :, i] = cv2.normalize(
            inverse_img[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if padding == True:
            # crop padded image back to original size
            output_img[:, :, i] = inverse_img[image_padding:-
                                              image_padding, image_padding:-image_padding, i]
        else:
            output_img[:, :, i] = inverse_img[:, :, i]

    # do some color modification to make the output look more close to original input
    output_img = brightness_contrast(output_img, brightness, contrast)

    return output_img


def RL_deconv(origin_img, PSF, iteration, brightness=50, contrast=100):

    origin_img, PSF = np.float32(origin_img), np.float32(PSF)

    # 初始化等一下會用到的東西
    est_conv = np.float32(
        np.zeros((origin_img.shape[0], origin_img.shape[1], 3)))
    relative_blur = np.float32(
        np.zeros((origin_img.shape[0], origin_img.shape[1], 3)))
    error_est = np.float32(
        np.zeros((origin_img.shape[0], origin_img.shape[1], 3)))

    # initial estimate
    latent_est = 0.5 * \
        np.float32(np.ones((origin_img.shape[0], origin_img.shape[1], 3)))
    # spatially reversed psf
    PSF_HAT = np.flipud(np.fliplr(PSF))

    # do computation on each BGR channel individually
    for i in range(3):
        for j in range(iteration):
            est_conv[:, :, i] = cv2.filter2D(latent_est[:, :, i], cv2.CV_32F,
                                             PSF, None, (-1, -1), 0, cv2.BORDER_REPLICATE)
            relative_blur[:, :, i] = origin_img[:, :, i] / est_conv[:, :, i]
            error_est[:, :, i] = cv2.filter2D(relative_blur[:, :, i], cv2.CV_32F,
                                              PSF_HAT, None, (-1, -1), 0, cv2.BORDER_REPLICATE)
            latent_est[:, :, i] = latent_est[:, :, i] * error_est[:, :, i]

        latent_est[:, :, i] = cv2.normalize(
            latent_est[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    output_img = latent_est

    # do some color modification to make the output look more close to original input
    output_img = brightness_contrast(output_img, brightness, contrast)

    return output_img


if __name__ == "__main__":
    # read in the two images
    pc_img = cv2.imread(
        "./images/Motion blur_1_3024x4032.jpg", cv2.IMREAD_COLOR)
    pc_img = np.asarray(pc_img)
    pc_img = blur_img(pc_img, 0.9)  # 稍微去噪

    duck_img = cv2.imread("./images/Donald_Duck.jpeg", cv2.IMREAD_COLOR)
    duck_img = np.asarray(duck_img)
    duck_img = blur_img(duck_img, 0.9)  # 稍微去噪

    # --------------------------------------------------------------------------------------------
    # guess keyboard motion blur function
    PSF = motion_kernel(angle=115, d=15)

    # apply wiener filter on keyboard
    # pc_wiener = wiener_filter(pc_img, PSF, padding=True, epsilon=0.01, K=30, b_gamma=1.2,
    #                           g_gamma=1.05, r_gamma=1.0, brightness=80, contrast=500)
    # plt.title('keyboard_wiener_filter')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((pc_wiener).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/key_wiener_filter.jpg',
    #             bbox_inches='tight', dpi=800)
    # plt.show()

    # apply inverse filter on keyboard
    # pc_inverse = inverse_filter(pc_img, PSF, epsilon=0.01, padding=True, n=8,
    #                             d0=158, b_gamma=1.2, g_gamma=1.05, r_gamma=1.0, brightness=250, contrast=300)
    # plt.title('keyboard_inverse_filter')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((pc_inverse).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/key_inverse_filter.jpg',
    #             bbox_inches='tight', dpi=800)
    # plt.show()

    # apply constrained least squares filter on keyboard
    # pc_CLS = CLS_filter(pc_img, PSF, padding=True, epsilon=0.01,
    #                     gamma=100, brightness=0, contrast=150)
    # plt.title('keyboard_CLS_filter')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((pc_CLS).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/key_CLS_filter.jpg',
    #             bbox_inches='tight', dpi=800)
    # plt.show()

    # apply Richardson–Lucy deconvolution on keyboard
    pc_RL = RL_deconv(pc_img, PSF, iteration=21, brightness=150, contrast=150)
    plt.title('keyboard_RL_deconv')
    plt.axis('off')
    plt.imshow(cv2.cvtColor((pc_RL).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.savefig('./images/key_RL_deconv.jpg', bbox_inches='tight', dpi=800)
    plt.show()

    # --------------------------------------------------------------------------------------------
    # guess duck motion blur function
    PSF_duck = motion_function(
        duck_img.shape[0], duck_img.shape[1], angle=-1, dist=80)

    # apply wiener filter on duck
    duck_wiener = wiener_filter(duck_img, PSF_duck, padding=False, epsilon=0.01, K=300,
                                brightness=80, contrast=150)
    plt.title('duck_wiener_filter')
    plt.axis('off')
    plt.imshow(cv2.cvtColor((duck_wiener).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.savefig('./images/duck_wiener_filter.jpg',
                bbox_inches='tight', dpi=800)
    plt.show()

    # apply inverse filter on duck
    # duck_inverse = inverse_filter(duck_img, PSF_duck, epsilon=0.0001, padding=False, n=70,
    #                               d0=8, brightness=50, contrast=100)
    # plt.title('duck_inverse_filter')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((duck_inverse).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/duck_inverse_filter.jpg',
    #             bbox_inches='tight', dpi=800)
    # plt.show()

    # apply constrained least squares filter on duck
    # duck_CLS = CLS_filter(duck_img, PSF_duck, padding=False,
    #                       epsilon=0.01, gamma=20000, brightness=-10, contrast=10)
    # plt.title('duck_CLS_filter')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((duck_CLS).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/duck_CLS_filter.jpg',
    #             bbox_inches='tight', dpi=800)
    # plt.show()

    # apply Richardson–Lucy deconvolution on duck
    # duck_RL = RL_deconv(duck_img, PSF_duck, iteration=7,
    #                     brightness=0, contrast=70)
    # plt.title('duck_RL_deconv')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor((duck_RL).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.savefig('./images/duck_RL_deconv.jpg', bbox_inches='tight', dpi=800)
    # plt.show()
