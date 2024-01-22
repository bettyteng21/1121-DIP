import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_two_histo(origin_img, processed_img, process_title):
    file_name = "./images/1_comparison_steak.jpg"

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

def contrast_limited_AHE(img):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    l, a, b = cv2.split(img)

    l = clahe.apply(l)
    enhanced_img = cv2.merge((l,a,b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    return enhanced_img

def unsharp_mask(img, k):

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    median_v = cv2.medianBlur(v, 13)

    # output = (1+k)*origin - k*blurred
    v = cv2.addWeighted(v, 1+k, median_v, -k, 0)
    enhanced_img = cv2.merge((h,s,v))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_HSV2BGR)

    return enhanced_img

def histogram_equalize(img):

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    v=cv2.equalizeHist(v)
    enhanced_img = cv2.merge((h,s,v))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_HSV2BGR)

    return enhanced_img

def gamma_transformation(origin_img, gamma):
    origin_img = origin_img.astype(np.uint8)

    # 把對應的值畫成table，之後直接查表
    gamma_table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    output_img = cv2.LUT(origin_img, gamma_table)

    return output_img

def enhance_meaty_part(img, gamma):
    b, g, r = cv2.split(img)
    r = gamma_transformation(r, gamma)
    enhanced_img = cv2.merge((b, g, r))

    return enhanced_img

def sharpen(img):

    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]]) 

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    v = cv2.filter2D(v, -1, kernel) 

    enhanced_img = cv2.merge((h,s,v))
    enhanced_img = cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return enhanced_img

def brightness_contrast(img, brightness, contrast):
    # 調整圖片的brightness和contrast
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    v = v * (contrast/127+1) - contrast + brightness
    v = np.clip(v, 0, 255).astype(np.uint8)

    enhanced_img = cv2.merge((h,s,v))
    enhanced_img = cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return enhanced_img


if __name__ == "__main__":
    origin_img = cv2.imread(
        "./images/steak.jpg", cv2.IMREAD_COLOR)
    origin_img = np.float32(origin_img)

    # sharpen img with a 3*3 kernel
    enhanced_img = sharpen(origin_img)

    # histogram equalization or clahe
    # enhanced_img = contrast_limited_AHE(enhanced_img)
    enhanced_img = histogram_equalize(enhanced_img)

    # unsharp mask
    enhanced_img = unsharp_mask(enhanced_img, 0.1)

    # adjust brightness and contrast
    enhanced_img = brightness_contrast(enhanced_img, 5, 15)

    # add a reddish tone to the image to make the steak more meaty
    enhanced_img = enhance_meaty_part(enhanced_img, 0.85)

    plot_two_histo(origin_img.astype(np.uint8), enhanced_img.astype(np.uint8), "enhanced_steak")
    cv2.imwrite('./images/1_steak.jpg', enhanced_img)