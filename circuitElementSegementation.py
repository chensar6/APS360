# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Image processing library - OpenCV

data_path = "CGHD-1152"

def cropAndResize(data_path, start_num, end_num):
    images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
    image_names = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

    images.sort()
    image_names.sort()
    # print(images)
    images.sort()
    image_names.sort()
    # print(images)

    # Plot the images in rows of 9
    for i in range(start_num, end_num, 9):
    # Figsize=(width in inches, length in inches)
        fig, ax = plt.subplots(1, 9, figsize=(60, 40))

    # Indicator to tell us how many more rows need to be printed
    # There are a total of 128 rows
        # print("Row: ", (i//9) +1)

        for j in range(9):
            if i+j >= len(images):
                ax[j].axis('off')
                continue

            # Load the image and convert to RGB color space
            # Note: OpenCV loads images in BGR color space so we need to convert to RGB
            curr_img = cv2.imread(images[i+j])
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            # Testing out image processing stuff
            edges = cv2.Canny(curr_img, 100, 150)
            
            # Padding images
            # Get the dimensions of the image
            height, width = edges.shape

            # Find the greater dimension
            greater_dim = max(height, width)

            # Create a blank image with the greater dimension
            padded_img = np.ones((greater_dim, greater_dim), dtype=np.uint8) * 0

            # Calculate the offsets for the original image
            x_offset = (greater_dim - width) // 2
            y_offset = (greater_dim - height) // 2

            # Copy the original image onto the padded image with the offset
            padded_img[y_offset:y_offset+height, x_offset:x_offset+width] = edges

            # Resize to 1024*1024
            resized_padded_img = cv2.resize(padded_img, (1024, 1024), cv2.INTER_NEAREST)

            ax[j].imshow(resized_padded_img)

            # Remove all y-axis ticks and labels
            ax[j].yaxis.set_ticks([])
            ax[j].yaxis.set_ticklabels([])
            ax[j].xaxis.set_ticks([])
            ax[j].xaxis.set_ticklabels([])

            # Remove the top and right spines
            ax[j].spines['top'].set_visible(False)
            ax[j].spines['right'].set_visible(False)

            ax[j].set_xlabel(image_names[i+j])

    plt.show()

cropAndResize(data_path, 0, 18)