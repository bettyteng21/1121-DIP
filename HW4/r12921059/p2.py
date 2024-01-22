import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_two_histo(origin_img, processed_img, process_title):
    file_name = "./images/2_comparison_cell.jpg"

    plt.subplot(1, 2, 1)
    plt.title('origin_img')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title(process_title)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    plt.savefig(file_name, bbox_inches='tight', dpi=800)
    plt.show()
    return

def brightness_contrast(img, brightness, contrast):
    # 調整圖片的brightness和contrast
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def do_img_mask(img, mask):
    # normalize mask img to 0~1
    mask = cv2.normalize(
        mask, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    # do mask on v channel
    v=mask*v

    masked_img = cv2.merge((h,s,v))
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)

    return masked_img

def method1(origin_img):
    img = origin_img
    # black_img = np.zeros((origin_img.shape[0], origin_img.shape[1]))
    
    # use g in BGR to make our mask image
    b, g, r = cv2.split(img)
    img = g.astype(np.uint8)
    img = brightness_contrast(img, 0, -50)
    img = cv2.medianBlur(img, 17) 
    img = np.array(255 * (img / 255) ** 1.21 , dtype='uint8')
    
    # set value=0 for those above 174
    ret, thresh = cv2.threshold(img, 174, 255, cv2.THRESH_BINARY_INV)

    # a large blurring filter to blur thresh img
    mask_img = cv2.medianBlur(thresh.astype(np.uint8), 111)

    # find contours and fill it with white => our mask img
    # contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     mask_img=cv2.drawContours(black_img, [contour], -1, (255,255,255), cv2.FILLED)

    # get masked image
    output_img = do_img_mask(origin_img, mask_img)

    return output_img


def method2(origin_img):
    
    img = origin_img
    black_img = np.zeros((origin_img.shape[0], origin_img.shape[1]))
    
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(img)
    img = s
    img = brightness_contrast(img,0, -50)
    img = cv2.medianBlur(img, 17) 
    img = np.array(255 * (img / 255) ** 1.25 , dtype='uint8')
    
    
    ret, thresh = cv2.threshold(img, 174, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.medianBlur(thresh.astype(np.uint8), 111)
    contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    mask_img=cv2.drawContours(black_img, contours, -1, (255,255,255), cv2.FILLED)

    output_img = do_img_mask(origin_img, mask_img)

    plt.title('aaa')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.imshow(thresh.astype(np.uint8), cmap='gray')
    plt.savefig("./images/2_cell2.jpg", bbox_inches='tight', dpi=800)
    plt.show()

    return 


if __name__ == "__main__":
    origin_img = cv2.imread(
        "./images/20-2851.tif", cv2.IMREAD_COLOR)
    origin_img = np.float32(origin_img)

    segmented_img = method1(origin_img)

    # same as method1 but different parameters
    # segmented_img = method2(origin_img)

    plot_two_histo(origin_img.astype(np.uint8), segmented_img.astype(np.uint8), "segmented_cell")
    cv2.imwrite('./images/2_cell.jpg', segmented_img)



