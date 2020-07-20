import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/test_img2.jpg')

# resize for quicker processing 
img_resized = cv2.resize(img, (900, 900))
# rgb version
img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# convert to grayscale for processing
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# otsu thresh
otsu,img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
img_otsu_inv = cv2.bitwise_not(img_otsu)

# adaptive threshold
img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
img_adap_gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# otsu and adaptive thres
otsu_and_adaptive = cv2.bitwise_and(img_otsu_inv,img_adap_gaus)
otsu_and_adaptive = cv2.dilate(otsu_and_adaptive, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
img_adap_gaus_inv = cv2.bitwise_not(img_adap_gaus)


# histogram
hist = cv2.calcHist([img_gray], [0], None, [256], [0,255])

# thresholded 
img_thres = np.ones_like(img_gray)
img_thres[True] = 255
low = 140
high = 190
img_thres[img_gray<low] = 0
img_thres[img_gray>high] = 0


# otsu and thres
otsu_and_thres = img_thres + img_otsu
otsu_and_thres = cv2.bitwise_not(otsu_and_thres)

# otsu and inv adaptive
img_adap_gaus_inv_and_otsu = cv2.bitwise_and(img_adap_gaus_inv, img_otsu)

# contours
contours, hierarchy = cv2.findContours(img_adap_gaus_inv_and_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_cnt = cv2.drawContours(img_resized_rgb.copy(), contours, -1, (0,255,0), 3)

#draw rect
img_rect = img_resized_rgb.copy()
egg_images = []
rect_coords = []
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100 or rect[2] > 400 or rect[3] > 400: 
        continue
    # print(cv2.contourArea(c))
    x,y,w,h = rect
    egg_images.append(img_gray[y:y+h, x:x+w])
    rect_coords.append(rect)
    cv2.rectangle(img_rect,(x,y),(x+w,y+h),(255,0,0),2)

subplots = [img_gray, img_otsu, img_adap_gaus, otsu_and_adaptive, img_adap_gaus_inv_and_otsu, img_adap_gaus_inv, img_cnt, img_rect]
subplot_names = ["img_gray", "img_otsu", "img_adap_gaus", "otsu_and_adaptive", "inv adap gaus and otsu", "img_adap_gaus_inv", "img_cnt", "img_rect"]

# plt.figure('analysis')
# for index, plots in enumerate(zip(subplots, subplot_names)):
#     pic, pic_name = plots
#     plt.subplot('24{0}'.format(index+1))
#     plt.imshow(pic, cmap='gray')
#     plt.title(pic_name)

def egg_transform(egg, style):
    # otsu
    if style == 1:
        otsu_thres, egg = cv2.threshold(egg, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        print(otsu_thres)
    #adaptive
    elif style == 2:
        egg = cv2.adaptiveThreshold(egg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # otsu and adaptive
    elif style == 3:
        _, egg_thresh = cv2.threshold(egg, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        egg_adaptive = cv2.adaptiveThreshold(egg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        egg = cv2.bitwise_and(egg_thresh, egg_adaptive)
    # manual thres
    elif style == 4:
        img_thres = np.ones_like(egg)
        img_thres[True] = 255
        low = 0
        high = 150
        img_thres[egg<low] = 0
        img_thres[egg>high] = 0
        egg = img_thres
    # contour ting
    elif style == 5:
        egg = cv2.adaptiveThreshold(egg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(egg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key=cv2.contourArea)[-1]
        print(c)
        egg = cv2.drawContours(egg.copy(), c, -1, 255, -1)
    elif style == 6:
        color_egg = cv2.cvtColor(egg, cv2.COLOR_GRAY2RGB)
        # egg = cv2.Canny(egg, 200, 50)
        egg = cv2.HoughCircles(egg, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=50, maxRadius=80)
        circles = np.uint16(np.around(egg))
        for x, y, r in circles[0,:]:
            egg = cv2.circle(color_egg, (x,y), r, (0,255,0), 3)

    return egg

egg_mask = list(map(lambda egg : egg_transform(egg, 6), egg_images))

plt.figure('processed eggs')
for index, rect in enumerate(zip(egg_mask, rect_coords)):
    egg_img, egg_coord = rect
    plt.subplot(3, 9 ,index+1)
    plt.imshow(egg_img, cmap='gray')
    plt.title('{0}'.format(index+1))
plt.show()
exit(0)
 
plt.figure('eggs')
for index, rect in enumerate(zip(list(map(lambda trans_egg, egg: cv2.bitwise_and(trans_egg, egg),
egg_mask, egg_images)), rect_coords)):
    egg_img, egg_coord = rect
    plt.subplot(3, 9 ,index+1)
    plt.imshow(egg_img, cmap='gray')
    plt.title('{0}'.format(index+1))

plt.figure('processed eggs')
for index, rect in enumerate(zip(egg_mask, rect_coords)):
    egg_img, egg_coord = rect
    plt.subplot(3, 9 ,index+1)
    plt.imshow(egg_img, cmap='gray')
    plt.title('{0}'.format(index+1))

# plt.figure('histogram')
# for index, rect in enumerate(zip(egg_images, rect_coords)):
#     egg_img, egg_coord = rect
#     hist = cv2.calcHist([egg_img], [0], None, [256], [0,255])
#     plt.subplot(3, 9 ,index+1)
#     plt.plot(hist)
#     plt.title('{0}'.format(index+1))


plt.show()

# cv2.imshow('rect', cv2.cvtColor(img_rect, cv2.COLOR_RGB2BGR))
# cv2.waitKey()
# cv2.destroyAllWindows()