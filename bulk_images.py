import os
import cv2
import numpy as np

def scalesss():
   
        resized_smaller_image = cv2.resize(smaller_image, None, fx=scale, fy=scale)

        # Rotate the smaller image
        rotation_matrix = cv2.getRotationMatrix2D((resized_smaller_image.shape[1] / 2, resized_smaller_image.shape[0] / 2), 1)
        rotated_smaller_image = cv2.warpAffine(resized_smaller_image, rotation_matrix, (resized_smaller_image.shape[1], resized_smaller_image.shape[0]))

        result = cv2.matchTemplate(larger_image, rotated_smaller_image, cv2.TM_CCOEFF_NORMED)

        # Define a threshold for similarity score
        threshold = 0.7

        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            bottom_right = (pt[0] + rotated_smaller_image.shape[1], pt[1] + rotated_smaller_image.shape[0])
            cv2.rectangle(larger_image, pt, bottom_right, (0, 255, 0), 2)

larger_image_path = r"E:\AWLx\Deep learning\others\dataset\illust_images\N_WM_257305_D_010006__AB_page_1.jpg"

larger_image_filename = os.path.basename(larger_image_path)  # Extract filename from the path
larger_image = cv2.imread(larger_image_path)

smaller_image_filenames = [
    "100ohm.JPG", 
    "coaxial.JPG", 
    "shield.JPG", 
    "star.JPG",
    "photosensitive 2.JPG",
 
    
]

scales = [0.5, 0.75, 1.0,1.25]

thresholds = result = {filename: 0.6 if filename == "shield_2.JPG" else (0.8 if "coaxial.JPG" in filename else 0.7) for filename in smaller_image_filenames}


colors = {
    "100ohm.JPG": (255, 0, 0),   
    "coaxial.JPG": (0, 255, 255),  
    "shield.JPG": (255, 0, 255),  
    "star.JPG": (0, 0, 255), 
    "photosensitive 2.JPG" :(255,255,0),
  
 
}

for smaller_image_filename in smaller_image_filenames:
    smaller_image_path = os.path.join(r"E:\check_img\data", smaller_image_filename)
    smaller_image = cv2.imread(smaller_image_path)

    for scale in scales:
        resized_smaller_image = cv2.resize(smaller_image, None, fx=scale, fy=scale)
        
        result = cv2.matchTemplate(larger_image, resized_smaller_image, cv2.TM_CCOEFF_NORMED)

        loc = np.where(result >= thresholds[smaller_image_filename])

        for pt in zip(*loc[::-1]):
            bottom_right = (pt[0] + resized_smaller_image.shape[1], pt[1] + resized_smaller_image.shape[0])
            
            color = colors[smaller_image_filename]
            
            cv2.rectangle(larger_image, pt, bottom_right, color, 2)

output_filename = f"Result_{os.path.splitext(larger_image_filename)[0]}.jpg"  
cv2.imwrite(output_filename, larger_image)  

cv2.imshow('Result', larger_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
                                                                                                                                                                                         
