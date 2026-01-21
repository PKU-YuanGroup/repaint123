import cv2
import numpy as np
import skimage.measure
from PIL import Image


def original_mask(img, kernel_size=5):
    if len(img.shape) == 3:
        mask = img[:, :, 3].cpu().numpy() >= 0.099
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = img
    return mask_uint8


def pool_mask(img, kernel_size=5):
    mask_uint8 = original_mask(img)
    mask_pooled = skimage.measure.block_reduce(mask_uint8, (8, 8), np.min)
    mask_pooled = mask_pooled.repeat(8, axis=0).repeat(8, axis=1)
    return mask_pooled


def erode_mask(img, kernel_size=5):
    mask_uint8 = original_mask(img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erode = cv2.erode(mask_uint8, kernel, iterations=1)
    return erode


def erode_pool_mask(img, kernel_size=5):
    erode = erode_mask(img, kernel_size=kernel_size)
    mask_pooled = skimage.measure.block_reduce(erode, (8, 8), np.min)
    mask_pooled = mask_pooled.repeat(8, axis=0).repeat(8, axis=1)
    return mask_pooled

def mean_pool_mask(img, kernel_size=5):
    mask_uint8 = original_mask(img)
    mask_pooled = skimage.measure.block_reduce(mask_uint8, (8, 8), np.mean)
    mask_pooled = mask_pooled.repeat(8, axis=0).repeat(8, axis=1)
    return mask_pooled

mask_proc_options = {
    0: original_mask,
    1: erode_mask,
    2: erode_pool_mask,
    3: pool_mask,
    4: mean_pool_mask,
}
mask_options = ["ori", "erode", "erode_pooled", "pooled_only", "mean_pooled"]


def inpaint_opencv(image_pil, mask_pil):
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    mask_cv2 = np.array(mask_pil)
    image_cv2 = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    return image_pil


def blend_img(img, mask_pil, kernel_size=8):
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mask = np.array(mask_pil)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_err = cv2.erode(mask, kernel, iterations=1)
    mask_blend = mask - mask_err

    img_blend = cv2.inpaint(img_np, mask_blend, kernel_size, cv2.INPAINT_TELEA)

    if isinstance(img, Image.Image):
        img_blend = Image.fromarray(cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB))

    return img_blend


def blend_mask(mask, kernel_size=8):
    mask_cv2 = np.array(mask)
    mask_cv2_bi = ((np.array(mask) != 0) * 255).astype(np.uint8)
    mask_cv2_er = ((np.array(mask) != 0) * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_cv2_er = cv2.erode(mask_cv2_er, kernel, iterations=1)
    mask_out = cv2.inpaint(
        mask_cv2, mask_cv2_er - mask_cv2_bi, kernel_size, cv2.INPAINT_TELEA
    )
    return Image.fromarray(mask_out)
