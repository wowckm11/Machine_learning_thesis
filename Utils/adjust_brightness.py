import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
for file in os.listdir('test_cases/images/'):
    path = f"test_cases/images/{file}"
    output_path = f"test_cases_normalised/images/{file}"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv.imwrite(output_path,cl1)