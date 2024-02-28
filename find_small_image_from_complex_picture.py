import cv2
import numpy as np

# Load the larger image
larger_image = cv2.imread(r"E:\illust_images\n_wm_257325_d_010001__ac_page_1.jpg")
smaller_image = cv2.imread(r'data\shield_2.JPG')




scales = [ 0.5, 0.75, 1.0, 1.25]
angles = [0, 90, 180, 270]


for scale in scales:
    for angle in angles:
        
        resized_smaller_image = cv2.resize(smaller_image, None, fx=scale, fy=scale)

        rotation_matrix = cv2.getRotationMatrix2D((resized_smaller_image.shape[1] / 2, resized_smaller_image.shape[0] / 2), angle, 1)
        rotated_smaller_image = cv2.warpAffine(resized_smaller_image, rotation_matrix, (resized_smaller_image.shape[1], resized_smaller_image.shape[0]))
        result = cv2.matchTemplate(larger_image, rotated_smaller_image, cv2.TM_CCOEFF_NORMED)


        threshold = 0.6

        loc = np.where(result >= threshold)


        for pt in zip(*loc[::-1]):
            bottom_right = (pt[0] + rotated_smaller_image.shape[1], pt[1] + rotated_smaller_image.shape[0])
            cv2.rectangle(larger_image, pt, bottom_right, (0, 255, 0), 2)



cv2.imshow('Result', larger_image)
cv2.imwrite('Result.jpg', larger_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
