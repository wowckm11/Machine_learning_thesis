import cv2
import matplotlib.pyplot as plt
import os 
import numpy as np
folder = 'chosen_models'
first_run = 0
for model_folder in os.listdir(folder):
    if first_run == 0:
        top_dir = f"{folder}/{model_folder}"
        image_list = []
        for image in os.listdir(top_dir):
            if image[0:6] in ('0.png','1.png','2.png','3.png','4.png'):
                image_list.append(image)
        
        for image in image_list:
            order_dict = {0:'0b86e07a-img-00041-00001.jpg',1:'0b9cf7fe-img-00005-00001.jpg',2:'0c80e5ad-img-00112-00002.jpg',3:'0dd3415d-img-00095-00001.jpg',4:'0e713f58-img-00087-00001.jpg'}

            #define paths
            original_image_path = f"val_images/{order_dict[int(image[0])]}"
            mask_image_path = f"saved_images/{image}"
            prediction_image_path = f"{top_dir}/pred_{image}"
            #load images
            original_image = cv2.imread(original_image_path)
            mask_image = cv2.imread(mask_image_path)
            prediction_image = cv2.imread(prediction_image_path)
            #resize
            original_image = cv2.resize(original_image, (800, 1200), None)
            mask_image = cv2.resize(mask_image, (800, 1200), None)
            prediction_image = cv2.resize(prediction_image, (800, 1200), None)
            #3-channel grey
            grey1 = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            grey2 = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
            grey3 = cv2.cvtColor(prediction_image, cv2.COLOR_RGB2GRAY)
            grey1_3_channel = cv2.cvtColor(grey1, cv2.COLOR_GRAY2BGR)
            grey2_3_channel = cv2.cvtColor(grey2, cv2.COLOR_GRAY2BGR)
            grey3_3_channel = cv2.cvtColor(grey3, cv2.COLOR_GRAY2BGR)
            #transform

            #numpy_vertical = np.vstack((image, grey_3_channel))
            numpy_horizontal = np.vstack((original_image, grey2_3_channel))
            numpy_horizontal = np.vstack((numpy_horizontal, grey3_3_channel))
            #numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
            #numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
            try:
                if int(image[0]) == 0:
                    vertical_stack = numpy_horizontal
                else:
                    vertical_stack = np.hstack((vertical_stack, numpy_horizontal))
            except NameError:
                vertical_stack = numpy_horizontal
        first_run=1
    else:
        top_dir = f"saved_models/{model_folder}"
        image_list = []
        for image in os.listdir(top_dir):
            if image[0:6] in ('0.png','1.png','2.png','3.png','4.png'):
                image_list.append(image)
        
        for image in image_list:
            #define paths
                
            prediction_image_path = f"{top_dir}/pred_{image}"
            #load images
            prediction_image = cv2.imread(prediction_image_path)
            #resize
            prediction_image = cv2.resize(prediction_image, (800, 1200), None)
            #3-channel grey
            grey3 = cv2.cvtColor(prediction_image, cv2.COLOR_RGB2GRAY)
            grey3_3_channel = cv2.cvtColor(grey3, cv2.COLOR_GRAY2BGR)
            #transform

            #numpy_vertical = np.vstack((image, grey_3_channel))
            #numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
            #numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
            try:
                if int(image[0]) == 0:
                    single_file_stack = grey3_3_channel
                else:
                    single_file_stack = np.concatenate((single_file_stack, grey3_3_channel), axis=1)
            except NameError:
                single_file_stack = grey3_3_channel
        vertical_stack = np.concatenate((vertical_stack, single_file_stack),axis=0)
    first_run = 1
plt.axis('off')
output = f"all_models_stack/{model_folder}_stack.png"
plt.imshow(vertical_stack)
plt.savefig(fname=output,format='png',bbox_inches='tight',transparent=True, pad_inches=0,dpi=300)
plt.close()
plt.show(block=False)
