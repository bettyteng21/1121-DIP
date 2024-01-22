import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def resize_rotate(img, scale, interpolation, degree):
    rad = math.radians(degree)  # degree to raduius
    h, w = img.shape[0], img.shape[1]
    resized_img = np.uint8(np.full((img.shape[0], img.shape[1]), 255))
    resized_h, resized_w = resized_img.shape[0], resized_img.shape[1]

    # 取resized img的中間當作原點
    resized_mid_h, resized_mid_w = round(
        ((resized_h+1)/2)-1), round(((resized_w+1)/2)-1)
    mid_h, mid_w = round(((h+1)/2)-1), round(((w+1)/2)-1)

    '''
    Note:
    要注意transfer的方向，要apply interpolation的話:
    需固定resized img的某點，去找該點對應到original img的哪點。

    如果方向相反(i.e.固定original img的某點，找它會在resized的哪裡)，
    做interpolation時，就無法取得周遭的pixel值(因為此時resized的pixel值尚未完全決定)
    '''
    for i in range(resized_h):
        for j in range(resized_w):
            # get和中間原點的相對位置
            y = i-resized_mid_h
            x = j-resized_mid_w

            # scaling
            origin_y = (y/scale)
            origin_x = (x/scale)

            # multiply rotaion matrix
            origin_y = (((-1)*origin_x*math.sin(rad)) +
                        (origin_y*math.cos(rad)))
            origin_x = (origin_x*math.cos(rad)) + (origin_y*math.sin(rad))

            # 從相對位置還原成pixel coordinate
            origin_y += mid_h
            origin_x += mid_w

            # apply different interpolation methods
            if 0 <= origin_x < w and 0 <= origin_y < h and origin_x >= 0 and origin_y >= 0:
                if interpolation == "nn":
                    resized_img[j, i] = nearest_neighbor(
                        img, origin_x, origin_y)
                elif interpolation == "bicubic":
                    resized_img[j, i] = bicubic(img, origin_x, origin_y)
                else:
                    # 剩下都預設是bilinear
                    resized_img[j, i] = bilinear(img, origin_x, origin_y)

    return resized_img


def nearest_neighbor(img, x, y):
    floor_x = math.floor(x)
    floor_y = math.floor(y)

    # get the min distance between target (x, y) and the four neighbours
    d = [
        math.dist([floor_x, floor_y+1], [x, y]),
        math.dist([floor_x+1, floor_y+1], [x, y]),
        math.dist([floor_x, floor_y], [x, y]),
        math.dist([floor_x+1, floor_y], [x, y])
    ]
    index = d.index(min(d))

    if (floor_x+1) < img.shape[1] and (floor_y+1) < img.shape[0]:
        # 正常情況
        if index == 0:
            value = img[floor_x, floor_y+1]
        elif index == 1:
            value = img[floor_x+1, floor_y+1]
        elif index == 2:
            value = img[floor_x, floor_y]
        elif index == 3:
            value = img[floor_x+1, floor_y]
        else:
            value = 255
    else:
        # 邊界pixel另外處理
        if (floor_x+1) >= img.shape[1] and (floor_y+1) >= img.shape[0]:
            # 最右上角的那一個pixel
            value = img[floor_x, floor_y]
        elif (floor_x+1) >= img.shape[1]:
            # 最右col的pixels
            if math.dist([y], [floor_y]) < math.dist([y], [floor_y+1]):
                value = img[floor_x, floor_y]
            else:
                value = img[floor_x, floor_y+1]
        else:
            # 最上row的pixels
            if math.dist([x], [floor_x]) < math.dist([x], [floor_x+1]):
                value = img[floor_x, floor_y]
            else:
                value = img[floor_x+1, floor_y]

    return value


def bilinear(img, x, y):
    h, w = img.shape[0], img.shape[1]
    x1, y1 = math.floor(x), math.floor(y)
    x2, y2 = math.floor(x)+1, math.floor(y)+1

    # 邊緣pixel另外預處理
    if x1 == (w-1):
        x1 = w-2
        x2 = w-1
    if y1 == (h-1):
        y1 = h-2
        y2 = h-1

    R1 = ((x2-x)/(x2-x1))*img[x1, y1] + ((x-x1)/(x2-x1))*img[x2, y1]
    R2 = ((x2-x)/(x2-x1))*img[x1, y2] + ((x-x1)/(x2-x1))*img[x2, y2]

    value = ((y2-y)/(y2-y1))*R1 + ((y-y1)/(y2-y1))*R2

    return value


def bicubic_formula(x):
    x = abs(x)
    a = -0.75  # 設得和pytorch一樣，論文是-0.5，可自訂
    if x <= 1:
        weight = (a+2)*(x**3) - (a+3)*(x**2) + 1
    elif x <= 2:
        weight = a*(x**3) - 5*a*(x**2) + 8*a*x - 4*a
    else:
        weight = 0

    return weight


def bicubic(img, x, y):
    temp_x, temp_y = x-math.floor(x), y-math.floor(y)  # 最近的左上點和(x, y)距離
    dist_x = [temp_x+1, temp_x, 1-temp_x, 2-temp_x]
    dist_y = [temp_y+1, temp_y, 1-temp_y, 2-temp_y]

    # 邊緣pixels另外處理
    f_x, f_y = math.floor(x), math.floor(y)
    if ((f_x + 2) or (f_x + 1)) >= img.shape[1]:
        x -= 2
    if (f_x - 1) < 0:
        x += 1

    if ((f_y + 2) or (f_y + 1)) >= img.shape[0]:
        y -= 2
    if (f_y - 1) < 0:
        y += 1

    # 和(x, y)相鄰的16點
    neighbor = [[img[math.floor(x)-1, math.floor(y)-1],  img[math.floor(x), math.floor(y)-1],
                img[math.floor(x)+1, math.floor(y)-1], img[math.floor(x)+2, math.floor(y)-1]],
                [img[math.floor(x)-1, math.floor(y)], img[math.floor(x), math.floor(y)],
                img[math.floor(x)+1, math.floor(y)], img[math.floor(x)+2, math.floor(y)]],
                [img[math.floor(x)-1, math.floor(y)+1], img[math.floor(x), math.floor(y)+1],
                img[math.floor(x)+1, math.floor(y)+1], img[math.floor(x)+2, math.floor(y)+1]],
                [img[math.floor(x)-1, math.floor(y)+2], img[math.floor(x), math.floor(y)+2],
                img[math.floor(x)+1, math.floor(y)+2], img[math.floor(x)+2, math.floor(y)+2]]
                ]
    neighbor = np.array(neighbor)

    # 填入x和y的weight表(都是16*16)
    weight_x = np.zeros((4, 4))
    weight_y = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            weight_x[i, j] = bicubic_formula(dist_x[j])

    for m in range(4):
        for n in range(4):
            weight_y[n, m] = bicubic_formula(dist_y[n])

    # neighbor*weight_x*weight_y(都是對應項相乘)，value=這16個值的總和
    value = 0
    for p in range(4):
        for q in range(4):
            value += (neighbor[p, q]*weight_x[p, q]*weight_y[p, q])

    if (value > 255):
        value = 255
    if (value < 0):
        value = 0

    return value


def diff_compare(origin_img, rr_img, interpolation):
    impulse_img = np.zeros(
        (origin_img.shape[0]*origin_img.shape[1]), np.uint8)

    flag = 0
    for i in range(rr_img.shape[0]):
        for j in range(rr_img.shape[1]):
            impulse_img[flag] = (rr_img[i, j]-origin_img[i, j])
            flag = flag+1

    plt.title('{}'.format(interpolation))
    plt.bar(range(origin_img.shape[0] *
            origin_img.shape[1]), impulse_img)
    plt.savefig('./images/{}.png'.format(interpolation))

    return


if __name__ == "__main__":
    origin_img = cv2.imread("./images/T.png", 0)
    # cv2.imshow('origin', origin_img)

    # resize_rotate input:
    # nn: nearest neighbor interpolation
    # bilinear: bilinear interpolation
    # bicubic: bicubic interpolation
    resize_rotate_nn = resize_rotate(origin_img, 0.7, "nn", -15)
    # cv2.imshow('resize_nn', resize_rotate_nn)
    cv2.imwrite("./images/resize&rotate_nn.png", resize_rotate_nn)
    # diff_compare(origin_img, resize_rotate_nn, "nearest_neighbor_plot")

    resize_rotate_bilinear = resize_rotate(origin_img, 0.7, "bilinear", -15)
    # cv2.imshow('resize_bilinear', resize_rotate_bilinear)
    cv2.imwrite("./images/resize&rotate_bilinear.png", resize_rotate_bilinear)
    # diff_compare(origin_img, resize_rotate_bilinear, "bilinear_plot")

    resize_rotate_bicubic = resize_rotate(origin_img, 0.7, "bicubic", -15)
    # cv2.imshow('resize_bicubic', resize_rotate_bicubic)
    cv2.imwrite("./images/resize&rotate_bicubic.png", resize_rotate_bicubic)
    # diff_compare(origin_img, resize_rotate_bicubic, "bicubic_plot")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# --------------------------------------------------------------------------

# 直接call的版本
    # origin = cv2.imread("./images/T.png", 0)
    # M = cv2.getRotationMatrix2D(
    #     ((origin.shape[1])/2, origin.shape[0]/2), -15, 0.7)

    # nearest = cv2.warpAffine(
    #     origin, M, (origin.shape[1], origin.shape[0]), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
    # status_nearest = cv2.imwrite('./images/nearest.png', nearest)
    # resize_nearest = cv2.resize(nearest, (0, 0), fx=20, fy=20)
    # view_nearest = resize_nearest[1000: 2000, 1500: 2000]
    # cv2.imshow('nearest', view_nearest)

    # bilinear = cv2.warpAffine(
    #     origin, M, (origin.shape[1], origin.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    # status_bilinear = cv2.imwrite('./images/bilinear.png', bilinear)
    # resize_bilinear = cv2.resize(bilinear, (0, 0), fx=20, fy=20)
    # view_bilinear = resize_bilinear[1000: 2000, 1500: 2000]
    # cv2.imshow('bilinear', view_bilinear)

    # bicubic = cv2.warpAffine(
    #     origin, M, (origin.shape[1], origin.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    # status_bicubic = cv2.imwrite('./images/bicubic.png', bicubic)
    # resize_bicubic = cv2.resize(bicubic, (0, 0), fx=20, fy=20)
    # view_bicubic = resize_bicubic[1000: 2000, 1500: 2000]
    # cv2.imshow('bicubic', view_bicubic)

    # cv2.waitKey(0)
