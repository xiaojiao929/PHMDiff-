# The original code is licensed under MIT License, which is can be found at licenses/LICENSE_UVIT.txt.

import cv2
import os

def downsample_image(image, scale_factor):
    """Downsample the image by a given scale factor."""
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return downsampled_image

def save_image(image, count):
    """Save the downsampled image with a unique name."""
    filename = f"downsampled_{count}.jpg"
    cv2.imwrite(filename, image)

def generate_downsampled_images(image, scale_factor):
    """Downsample the image iteratively until its resolution is less than 100 pixels."""
    count = 0
    while image.shape[0] >= 100 and image.shape[1] >= 100:
        save_image(image, count)
        image = downsample_image(image, scale_factor)
        count += 1
    return image

if __name__ == "__main__":
    main()
