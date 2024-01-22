import numpy as np
import cv2
from matplotlib import pyplot as plt


def DFT(input):
    N = input.size
    V = np.zeros((N, N), dtype=np.complex128)
    for y in range(N):
        for x in range(N):
            V[y, x] = np.exp(-1j*2*np.pi*x*y/N)

    return input.dot(V)


def IDFT(input):
    N = input.size
    V = np.zeros((N, N), dtype=np.complex128)
    for y in range(N):
        for x in range(N):
            V[y, x] = np.exp(1j*2*np.pi*x*y/N)

    return (1/N)*input.dot(V)


def FFT(x):
    N = x.shape[1]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:
        temp_dft = np.zeros((x.shape[0], x.shape[1]), dtype=np.complex128)

        for i in range(x.shape[0]):
            temp_dft[i] = DFT(x[i, :])

        return temp_dft
    else:
        X_even = FFT(x[:, ::2])
        X_odd = FFT(x[:, 1::2])

        factor = np.zeros((x.shape[0], x.shape[1]), dtype=np.complex128)

        for i in range(x.shape[0]):
            factor[i] = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.hstack([X_even + np.multiply(factor[:, :int(N/2)], X_odd),
                          X_even + np.multiply(factor[:, int(N/2):], X_odd)])


def IFFT(x):
    N = x.shape[1]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:
        temp_dft = np.zeros((x.shape[0], x.shape[1]), dtype=np.complex128)

        for i in range(x.shape[0]):
            temp_dft[i] = IDFT(x[i, :])

        return temp_dft
    else:
        X_even = IFFT(x[:, ::2])
        X_odd = IFFT(x[:, 1::2])

        factor = np.zeros((x.shape[0], x.shape[1]), dtype=np.complex128)

        for i in range(x.shape[0]):
            factor[i] = np.exp(2j * np.pi * np.arange(N) / N)

        # 因為是遞迴且shape為2的次方，所以ifft每次return都多除2，就相當於公式裡的那個除N
        return np.hstack([(X_even + np.multiply(factor[:, :int(N/2)], X_odd))/2,
                          (X_even + np.multiply(factor[:, int(N/2):], X_odd))/2])


def FFT2D(img):
    # 2d相當於做兩次fft，分別在不同維度
    return FFT(FFT(img).T).T


def IFFT2D(img):
    return IFFT(IFFT(img).T).T


def FFT_SHIFT(img):
    '''
    目標: block 1, 3互換，block 2, 4互換

    1   2         3   4
             ->   
    4   3         2   1

    '''
    M, N = img.shape
    M = int(M/2)
    N = int(N/2)

    # 上半下半交換，再左半右半交換
    return np.vstack((np.hstack((img[M:, N:], img[M:, :N])), np.hstack((img[:M, N:], img[:M, :N]))))


def IFFT_SHIFT(img):
    '''
    目標: 還原成原始的block 1234

    3   4         1   2
             ->   
    2   1         4   3

    '''
    M, N = img.shape
    M = int(M/2)
    N = int(N/2)

    # 左半右半交換，再上半下半交換
    return np.hstack((np.vstack((img[M:, N:], img[:M, N:])), np.vstack((img[M:, :N], img[:M, :N]))))


def butterworth_filter(img, D0, n, uk, vk):
    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    Dm = np.sqrt((u - M//2 - uk)**2 + (v - N//2 - vk)**2)
    Dp = np.sqrt((u - M//2 + uk)**2 + (v - N//2 + vk)**2)
    HPF = (1 / (1 + (D0 / (Dm + 1e-6))**(2*n))) * \
        (1 / (1 + (D0 / (Dp + 1e-6))**(2*n)))

    return HPF


def crop(img, target_h, target_w):
    recover_img = np.zeros((target_h, target_w), np.uint8)
    # 裁切區域的 x 與 y 座標（左上角）
    x = int((img.shape[1]-recover_img.shape[1])/2)
    y = int((img.shape[0]-recover_img.shape[0])/2)

    # 裁切區域的長度與寬度
    w = recover_img.shape[1]
    h = recover_img.shape[0]

    # 裁切圖片
    recover_img = img[y:y+h, x:x+w]

    return recover_img


def normalize(img):
    # max取255，min取0
    return (img-img.min()) * (255/(img.max()-img.min()))


def border_wrap_pad(img, padded_h, padded_w):
    '''
    divide padded image into several blocks:
    X is the original img, block A~H is the boarder

      E |    D    | F
    --------------------
        | -  -  - |
      A | -  X  - | B
        | -  -  - |
    --------------------
      G |    C    | H

    '''
    img_h, img_w = img.shape[0], img.shape[1]
    padded_img = np.zeros((padded_w, padded_h), np.uint8)

    block_D = img[img_h-int((padded_h-img_h)/2):img_h, :]
    block_C = img[:(padded_h-img_h-block_D.shape[0]), :]
    block_DXC = np.concatenate([block_D, img, block_C], axis=0)
    # print("size D= {},  size C= {}, sizeDXC= {}".format(
    #     block_D.shape, block_C.shape, block_DXC.shape))

    block_E = img[img_h-int((padded_h-img_h)/2):img_h,
                  img_w-int((padded_w-img_w)/2):img_w]
    block_A = img[:, img_w-int((padded_w-img_w)/2):img_w]
    block_G = img[:(padded_h-img_h-block_D.shape[0]),
                  img_w-int((padded_w-img_w)/2):img_w]
    block_EAG = np.concatenate([block_E, block_A, block_G], axis=0)
    # print("size E= {},  size A= {}, size G= {}, , size EAG= {}".format(
    #     block_E.shape, block_A.shape, block_G.shape, block_EAG.shape))

    block_F = img[img_h-int((padded_h-img_h)/2):img_h,
                  :(padded_w-img_w-block_E.shape[1])]
    block_B = img[:, :(padded_w-img_w-block_E.shape[1])]
    block_H = img[:(padded_h-img_h-block_D.shape[0]),
                  :(padded_w-img_w-block_E.shape[1])]
    block_FBH = np.concatenate([block_F, block_B, block_H], axis=0)
    # print("size F= {},  size B= {}, size H= {}, , size FBH= {}".format(
    #     block_F.shape, block_B.shape, block_H.shape, block_FBH.shape))

    padded_img = np.concatenate([block_EAG, block_DXC, block_FBH], axis=1)

    return padded_img


def customized_pad(img, padded_h, padded_w):
    '''
    divide padded image into several blocks:
    X is the original img, block A~H is the boarder

      E |    D    | F
    --------------------
        | -  -  - |
      A | -  X  - | B
        | -  -  - |
    --------------------
      G |    C    | H


    針對astronaut_interference.tif這張圖做客製化的padding

    分析如下:
    block D: copy X的上半部直接貼上(非mirroring)，既可延伸柵欄線條，同時讓background保持黑色
    block C: 概念相同，copy X的下半部
    block A, B: 和前面wrap padding一樣
    block E, F: 分別取X左上角和右上角
    block G, H: 分別取X左下角和右下角

    '''
    img_h, img_w = img.shape[0], img.shape[1]
    padded_img = np.zeros((padded_h, padded_w), np.uint8)

    block_C = img[img_h-int((padded_h-img_h)/2):img_h, :]
    block_D = img[:(padded_h-img_h-block_C.shape[0]), :]
    block_DXC = np.concatenate([block_D, img, block_C], axis=0)
    # print("size D= {},  size C= {}, sizeDXC= {}".format(
    #     block_D.shape, block_C.shape, block_DXC.shape))

    block_G = img[img_h-int((padded_h-img_h)/2):img_h,
                  img_w-int((padded_w-img_w)/2):img_w]
    block_A = img[:, img_w-int((padded_w-img_w)/2):img_w]
    block_E = img[:(padded_h-img_h-block_C.shape[0]),
                  img_w-int((padded_w-img_w)/2):img_w]
    block_EAG = np.concatenate([block_E, block_A, block_G], axis=0)
    # print("size E= {},  size A= {}, size G= {}, , size EAG= {}".format(
    #     block_E.shape, block_A.shape, block_G.shape, block_EAG.shape))

    block_H = img[img_h-int((padded_h-img_h)/2):img_h,
                  :(padded_w-img_w-block_E.shape[1])]
    block_B = img[:, :(padded_w-img_w-block_E.shape[1])]
    block_F = img[:(padded_h-img_h-block_C.shape[0]),
                  :(padded_w-img_w-block_E.shape[1])]
    block_FBH = np.concatenate([block_F, block_B, block_H], axis=0)
    # print("size F= {},  size B= {}, size H= {}, , size FBH= {}".format(
    #     block_F.shape, block_B.shape, block_H.shape, block_FBH.shape))

    padded_img = np.concatenate([block_EAG, block_DXC, block_FBH], axis=1)

    return padded_img


if __name__ == "__main__":

    img = cv2.imread('./images/astronaut_interference.tif', 0)

    # 取客製化的padding
    padded_img = customized_pad(img, 1024, 1024)

    # 取標準的wrap padding
    # padded_img = border_wrap_pad(img, 1024, 1024)
    # cv2.imshow("result", padded_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # fourier transform
    f_shift = FFT_SHIFT(FFT2D(padded_img))

    # test between hand-crafted & numpy
    # target_fft = abs(np.fft.fftshift(np.fft.fft2(padded_img)))
    # is_close = np.allclose(abs(f_shift), target_fft)
    # print('FFT+FFT_SHIFT distance from numpy.fft+fftshift: ',
    #       np.linalg.norm(abs(f_shift)-target_fft))
    # print('is_close: ', is_close)

    # butterworth_filter(img, D0, n, uk, vk)
    HPF = butterworth_filter(padded_img, 25, 5, 27, 18)

    # fourier transform dot butterworth filter
    Gshift = f_shift * HPF

    # plt.subplot(2, 2, 1)
    # plt.title('Original(Padded)')
    # plt.imshow(padded_img, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.title('FFT2D')
    # plt.imshow(np.log(1+abs(f_shift)), cmap='gray')
    # plt.subplot(2, 2, 3)
    # plt.title('G_shift')
    # plt.imshow(20*np.log(np.abs(Gshift)), cmap='gray')

    G = IFFT_SHIFT(Gshift)

    # test between hand-crafted & numpy
    # target_G = np.fft.ifftshift(Gshift)
    # is_close = np.allclose(G, target_G)
    # print('IFFT_SHIFT distance from numpy.ifftshift: ',
    #       np.linalg.norm(G-target_G))
    # print('is_close: ', is_close)

    g = np.abs(IFFT2D(G))

    # test between hand-crafted & numpy
    # target_g = np.abs(np.fft.ifft2(G))
    # is_close = np.allclose(g, target_g)
    # print('IFFT distance from numpy.ifft: ', np.linalg.norm(g-target_g))
    # print('is_close: ', is_close)

    # plt.subplot(2, 2, 4)
    # plt.title('Result(Padded)')
    # plt.imshow(g, cmap='gray')
    # plt.show()

    recover_img = crop(g, img.shape[0], img.shape[1])
    recover_img = normalize(recover_img)
    # cv2.imshow("recover", recover_img)
    cv2.imwrite('./images/recover_filtered_img.png', recover_img)
    cv2.waitKey(0)

    # --------------------------------------------------------------------------

    # 直接call的版本
    # img = cv2.imread('./images/astronaut_interference.tif', 0)

    # # fourier transform
    # imgFloat32 = np.float32(img)
    # F = np.fft.fft2(img)
    # print(F.dtype)
    # Fshift = np.fft.fftshift(F)
    # print(Fshift.dtype)
    # magnitude_spectrum = 20*np.log(np.abs(Fshift))
    # # plt.imshow(20*np.log(np.abs(Fshift)), cmap='gray')
    # # plt.axis('off')
    # # plt.show()

    # HPF = butterworth_filter(img, 25, 5, 27, 18)
    # plt.imshow(HPF, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # # fourier transform dot butterworth filter
    # Gshift = Fshift * HPF
    # # plt.imshow(20*np.log(np.abs(Gshift)), cmap='gray')
    # # plt.axis('off')
    # # plt.show()

    # # inverse fourier transform
    # G = np.fft.ifftshift(Gshift)
    # g = np.abs(np.fft.ifft2(G))

    # recover_img = normalize(g)
    # cv2.imwrite('./images/filtered_img.png', recover_img)
    # plt.imshow(g, cmap='gray')
    # plt.axis('off')
    # plt.show()
