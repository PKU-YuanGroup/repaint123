import cv2
import numpy as np
import os


def create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def import_config_key(config, key, default=""):
    return config.get(key, default)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return max(fm, 1)


def listify_matrix(mat):
    return np.ndarray.tolist(mat.cpu().numpy())

def view_dep_prompt(prompt, angle, color=""):
    if color == "":
        base_prompt = f"A photo of a {prompt}"
    else:
        base_prompt = f"A photo of a {color} {prompt}"
    if angle <= 45 or angle >= 315:
        return f"{base_prompt}, front view"
    if 45 <= angle <= 135:
        return f"{base_prompt}, left view"
    if 135 <= angle <= 225:
        return f"{base_prompt}, back view"
    if 225 <= angle <= 315:
        return f"{base_prompt}, right view"
    return prompt