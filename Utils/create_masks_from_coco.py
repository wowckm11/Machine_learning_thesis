import json
import numpy as np
import cv2
import gc
import matplotlib.pyplot as plt
f = open('coco_new_normalised_brightness/result.json')
f = json.load(f)
pair = []
dict_of_image_id = {}
filename_dict = {}
for content in f['annotations']:
    dict_of_image_id[content['image_id']]= []
for content in f['annotations']:
    dict_of_image_id[content['image_id']].append(content['segmentation'])
for content in f['images']:
    filename_dict[content['id']] = content['file_name']
print(filename_dict)
all_image_polygons=[]
for image in dict_of_image_id.keys():
    #print(f"i expect {len(dict_of_image_id[image])} polygons for photo {image}")
    for i in (range(len(dict_of_image_id[image]))):
        #print(f"i create polygon {i+1}/{len(dict_of_image_id[image])}")
        for j in range(len(dict_of_image_id[image][i][0])):
            try:
                if dict_of_image_id[image][i][0][2*j:2*(j+1)] != []:
                    pair.append(dict_of_image_id[image][i][0][2*j:2*(j+1)])
            except:
                print(f"i was going through image nr{i}, pair slice {2*j}to {2*(j+1)} and failed to add it")
        all_image_polygons.append(pair)
        pair = []
    dict_of_image_id[image] = all_image_polygons
    all_image_polygons = []
for image in dict_of_image_id.keys():
    filename = filename_dict[image]
    output =  f"mask/{filename_dict[image][7:]}"
    mask = cv2.imread(f'coco_new/{filename}')
    #mask[ : ] = (0,0,0)
    for polygon in dict_of_image_id[image]:
        polygons = [np.array(polygon, dtype=np.int32)]
        mask = cv2.fillPoly(mask, polygons,(255,255,255))
    cv2.imwrite(output,mask)
    print(f"image {output} saved, it was nr {image}")
    print(gc.collect())
