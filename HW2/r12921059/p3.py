import numpy as np
import cv2
from matplotlib import pyplot as plt


def fourier_spectrum(origin_img):
    imgFloat32 = np.float32(origin_img)
    F = np.fft.fft2(imgFloat32)

    Fshift = np.fft.fftshift(F)
    output_img = 20*np.log(np.abs(Fshift))

    return output_img


def pad_image(image, rows, cols, odd_symm=False):
    image_pad = np.zeros((rows, cols))

    # 把kernel pad到圖片中央
    if odd_symm == False:
        mid_row = rows // 2
        mid_col = cols // 2
        image_pad[(mid_row-1):(mid_row+2), (mid_col-1):(mid_col+2)] = image

    else:
        mid_row = rows // 2
        mid_col = cols // 2
        image_pad[(mid_row-2):(mid_row+2), (mid_col-2):(mid_col+2)] = image
    return image_pad


def conv2_fft(origin_img, kernel, odd_symm):

    h, w = origin_img.shape[0], origin_img.shape[1]

    kernel_padded = pad_image(kernel, h, w, odd_symm)

    fft_origin =np.fft.fft2(origin_img)
    fft_kernel = np.fft.fft2(kernel_padded)

    fft_kernel.real = 0

    out_freq = (fft_origin*fft_kernel)

    output_img = np.fft.ifftshift(np.fft.ifft2(out_freq)).real

    return output_img


def odd_vertical_sobel(origin_img, kernel):

    '''
    size= 4*4
    filter2D 要input的kernel
    odd_sobel_kernel = np.array([[0,  0, 0,  0],
                                 [0, -1, 0, 1],
                                 [0, -2, 0, 2],
                                 [0, -1, 0, 1]])
    '''

    kernel = (-1)*kernel
    output_img = cv2.filter2D(origin_img, cv2.CV_64F,
                              kernel, None, (-1, -1), 0, cv2.BORDER_CONSTANT)

    return output_img


def vertical_sobel(origin_img):
    '''
    xorder = 0, yorder = 1, ksize = 3

    Sobel kernel:
    [[-1,  0,  1],
     [-2,  0,  2],
     [-1,  0,  1]]
    '''

    output_img = cv2.Sobel(origin_img, cv2.CV_64F, 1, 0, ksize=3)

    return output_img


if __name__ == "__main__":
    origin_img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
    origin_img = np.asarray(origin_img)

    # show fourier spectrum ------------------------------------------------------------------
    plt.title('(a) Fourier Spectrum')
    plt.imshow(fourier_spectrum(origin_img), cmap='gray')
    plt.axis('off')
    plt.savefig('./images/3a_fourier_spectrum.jpg', bbox_inches='tight', dpi=500)
    plt.show()

    # Enforce odd symmetry --------------------------------------------------------------------
    odd_sobel_kernel = np.array([[0, 0, 0,  0],
                                 [0, 1, 0, -1] ,
                                 [0, 2, 0, -2],
                                 [0, 1, 0, -1]])

    # Zero-padding
    odd_pad_h = origin_img.shape[0] + odd_sobel_kernel.shape[0] - 1  # 1134+4-1 = 1137
    odd_pad_w = origin_img.shape[1] + odd_sobel_kernel.shape[1] - 1  # 1360+4-1 = 1363

    odd_pad_orgin = np.zeros((odd_pad_h+1, odd_pad_w+1)) 
    odd_pad_orgin[:origin_img.shape[0], :origin_img.shape[1]] = origin_img

    plt.subplot(2, 2, 1)
    plt.title('Padded origin')
    plt.imshow(odd_pad_orgin, cmap='gray')
    plt.axis('off')

    odd_freq_filter_img= conv2_fft(odd_pad_orgin, odd_sobel_kernel, odd_symm=True)
    plt.subplot(2, 2, 2)
    plt.title('freq domain vertical sobel')
    plt.imshow(odd_freq_filter_img, cmap='gray')
    plt.axis('off')

    odd_spatial_filter_img= odd_vertical_sobel(odd_pad_orgin, odd_sobel_kernel)
    plt.subplot(2, 2, 3)
    plt.title('spatial domain vertical sobel')
    plt.imshow(odd_spatial_filter_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Diff')
    plt.imshow(np.abs(odd_freq_filter_img - odd_spatial_filter_img), cmap='gray')
    plt.axis('off')

    plt.suptitle('(c)(d) Enforce Odd Symmetry', fontsize= 20, fontweight='bold')
    plt.savefig('./images/3cd_odd_symm_result.jpg', bbox_inches='tight', dpi=500)
    plt.show()

    # w/o odd symmetry -----------------------------------------------------------------------
    sobel_kernel = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])
    
    # Zero-padding
    pad_h = origin_img.shape[0] + sobel_kernel.shape[0] - 1  
    pad_w = origin_img.shape[1] + sobel_kernel.shape[1] - 1  

    pad_origin = np.zeros((pad_h, pad_w))
    pad_origin[:origin_img.shape[0], :origin_img.shape[1]] = origin_img

    plt.subplot(2, 2, 1)
    plt.title('Padded origin')
    plt.imshow(pad_origin, cmap='gray')
    plt.axis('off')

    freq_filter_img = conv2_fft(pad_origin, sobel_kernel,odd_symm=False)
    plt.subplot(2, 2, 2)
    plt.title('freq domain vertical sobel')
    plt.imshow(freq_filter_img, cmap='gray')
    plt.axis('off')

    spatial_filter_img = vertical_sobel(pad_origin)
    plt.subplot(2, 2, 3)
    plt.title('spatial domain vertical sobel')
    plt.imshow(spatial_filter_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Diff')
    plt.imshow(np.abs(freq_filter_img - spatial_filter_img), cmap='gray')
    plt.axis('off')

    plt.suptitle('(e) Without Odd Symmetry', fontsize= 20, fontweight='bold')
    plt.savefig('./images/3e_wo_odd_symm_reasult.jpg', bbox_inches='tight', dpi=500)
    plt.show()

    

    