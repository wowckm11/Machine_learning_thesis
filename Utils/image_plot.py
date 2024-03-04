import os
import matplotlib.pyplot as plt
import cv2
import random
IMAGES_PATH = str("coco_new_normalised_brightness")
ANNOTATIONS_PATH = str("coco_new_normalised_brightness/result.json")



filename = f"{IMAGES_PATH}/images/0b9cf7fe-img-00005-00001.jpg"


image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #conver image to RGB in order to use in matplotlib.
rows, cols = image.shape[:2]
aug_img_V_Flip = cv2.flip(image, 1)
for i in range(6):
    output = f"augmented_photo/rotate{i}.jpg"
    Cx , Cy = rows, cols
    rand_angle = random.randint(-45,45)
    M = cv2.getRotationMatrix2D((Cy//2, Cx//2),rand_angle ,1)
    aug_img_final = cv2.warpAffine(image, M, (cols, rows))
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(aug_img_final)
    plt.savefig(fname=output,format='png',bbox_inches='tight',transparent=True, pad_inches=0,dpi=300)
    plt.show()