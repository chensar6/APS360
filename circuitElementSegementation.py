# Imports
import os
import gc
import json
import matplotlib.pyplot as plt
import numpy as np

# Image processing library - OpenCV
import cv2  

data_path = "CGHD-1152"

def preprocess():
    # Get all the circuit images - (element 1: full path to circuit image file, element 2: just the name of the circuit image file)
    images = [(os.path.join(data_path, f), f) for f in os.listdir(data_path) if (f.endswith('.jpg'))]
    images.sort()     # may come in handy later

    # Load dictionary that describes how each circuit will be preprocessed
    circuit_to_ppp = dict()

    with open('circuit_preprocessing_parameters.json', 'r') as f:
        json_str = f.read()

    circuit_to_ppp = json.loads(json_str)

    # Create set to track preprocessing progress
    try:
        processed_circuits = set(os.listdir(os.getcwd() + "\\Processed"))
    except:
        os.mkdir("Processed")
        processed_circuits = set()

    # Loop through images, preprocess and save to folder
    for (circuit_file, circuit_name) in images:
        # Clear ram so that we don't encounter memory issues
        gc.collect()

        # Check that we didn't already process this image - save on computations
        if os.path.exists("Processed/" + circuit_name):
            continue

        # Print progress check
        print("Currently processing file {0}/1152...".format(len(processed_circuits) + 1))
        processed_circuits.add(circuit_name)

        # Get the ID of the current circuit to access its preprocessing parameters from dict
        circuit_id = circuit_name.split("_")[0] + "_"

        # Load the image and convert to grayscale color space
        # Note: OpenCV loads images in BGR color space
        curr_img = cv2.imread(circuit_file)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to get rid of background colors, creases, and only remain with actual circuit
        thresholded_img = cv2.adaptiveThreshold(curr_img, 
                                                255, 
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY,
                                                circuit_to_ppp[circuit_id][0],
                                                circuit_to_ppp[circuit_id][1])
        
        # Apply median blur to get rid of "salt and pepper" noise
        processed_img = cv2.medianBlur(thresholded_img, circuit_to_ppp[circuit_id][2])

        # Invert the image so that the circuits are labelled as 255 while background is 0
        processed_img = cv2.bitwise_not(processed_img)

        # Normalize so that instead of 0 & 255, values are 0 & 1, also float
        processed_img = processed_img / 255.0

        # Save processed image if not processed before
        cv2.imwrite("Processed/" + circuit_name, processed_img)

        # If you want to see what the final processed image looks like, uncomment this code
        #plt.imshow(processed_img)
        #plt.show()
    
    print("\nDone Processing!")

if __name__ == "__main__":
    preprocess()