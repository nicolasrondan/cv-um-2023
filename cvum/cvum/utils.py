import cv2
import numpy as np
from typing import Tuple

def imread_rgb(image_path:str) -> np.ndarray:
    """ OpenCV Wrapper to read image in RGB instead of BGR

    Args:
        image_path: image path from where to read image.

    Returns:
        RGB image read from the selected path.
    """


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def add_snp_noise(image:np.ndarray, amount=0.005) -> np.ndarray:
    """ Add Salt and Pepper noise to an image

    Args:
        image: 2D Grayscale input image to add noise
        amount: percentage of pixels to distort with salt and pepper noise

    Returns:
        A new image with salt and pepper noise added using the selected amount

    """

    s_vs_p = 0.5
    output_image = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
             for i in image.shape]
    output_image[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
             for i in image.shape]
    output_image[tuple(coords)] = 0
    
    #print(coords.shape)
    
    return output_image

def correlationdot_2D(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:

    """ Two-dimensional cross-correlation between the input image and the desired kernel.

    Args:
        image: 2D Grayscale input image to be filtered.
        kernel: 2D filter that will be applied to input image.

    Returns:
        The image resulting of the cross-correlation between input image and kernel.

    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if image.dtype == np.uint8:
        image = image / 255. 
    
    m = image_height - kernel_height + 1; 
    n = image_width - kernel_width + 1; 
    correlation_result = np.zeros([m,n])
    for y in range(0,m):
        for x in range(0,n):
         im_patch = image[y:y+kernel_height, x:x+kernel_width]
         correlation_result[y,x] = im_patch.ravel().T.dot(kernel.ravel())

    return correlation_result

def calculate_image_gradient(image_grayscale: np.ndarray) -> Tuple[np.ndarray]:
    
    """ Calculates de image gradient magnitude and angle given a grayscale image
    
    Args:
        image_grayscale: 2D Grayscale input image.

    Returns:
        A Tuple containing the image gradient magnitude matrix and the image gradiente direction matrix

    """

    sobel_x = cv2.Sobel(image_grayscale,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(image_grayscale,cv2.CV_64F,0,1,ksize=5)

    g_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    g_direction = np.arctan(np.divide(sobel_y,sobel_x))

    return g_magnitude, g_direction

