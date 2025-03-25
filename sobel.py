import numpy as np
import math
from PIL import Image
import os

def sobel_edge_detection(input_file, 
                         low_output_file, 
                         high_output_file, 
                         mag_output_file, 
                         low_threshold, 
                         high_threshold):
    # Initialize the masks
    maskx = np.array([[-1,  0,  1],
                      [-2,  0,  2],
                      [-1,  0,  1]])
    masky = np.array([[ 1,  2,  1],
                      [ 0,  0,  0],
                      [-1, -2, -1]])

    # Read the input PGM image (convert to grayscale just in case)
    img = Image.open(input_file).convert('L')
    pic = np.array(img, dtype=np.int32)

    rows, cols = pic.shape

    # Initialize output arrays
    outpicx = np.zeros((rows, cols))
    outpicy = np.zeros((rows, cols))
    ival = np.zeros((rows, cols))

    # Apply the Sobel operator
    mr = 1
    for i in range(mr, rows - mr):
        for j in range(mr, cols - mr):
            sum1 = 0
            sum2 = 0
            for p in range(-mr, mr + 1):
                for q in range(-mr, mr + 1):
                    sum1 += pic[i + p, j + q] * maskx[p + mr, q + mr]
                    sum2 += pic[i + p, j + q] * masky[p + mr, q + mr]
            outpicx[i, j] = sum1
            outpicy[i, j] = sum2

    # Compute gradient magnitude
    maxival = 0
    for i in range(mr, rows - mr):
        for j in range(mr, cols - mr):
            ival[i, j] = math.sqrt(outpicx[i, j]**2 + outpicy[i, j]**2)
            if ival[i, j] > maxival:
                maxival = ival[i, j]

    # Normalize to 0–255
    ival = (ival / maxival) * 255

    # --------
    # 1) Save the raw gradient magnitude as an image
    # --------
    mag_output_img = Image.fromarray(ival.astype(np.uint8))
    mag_output_img.save(mag_output_file)

    # Apply low threshold
    low_thresholded = ival.copy()
    low_thresholded[low_thresholded < low_threshold] = 0

    # Apply high threshold
    high_thresholded = ival.copy()
    high_thresholded[high_thresholded < high_threshold] = 0

    # 2) Save the low-threshold image
    low_output_img = Image.fromarray(low_thresholded.astype(np.uint8))
    low_output_img.save(low_output_file)

    # 3) Save the high-threshold image
    high_output_img = Image.fromarray(high_thresholded.astype(np.uint8))
    high_output_img.save(high_output_file)

if __name__ == "__main__":
    # Hardcoded file paths and thresholds
    input_file       = "C:/Users/leema/Downloads/HW1/HW1/garb34.pgm"
    low_output_file  = "C:/Users/leema/Downloads/HW1/HW1/sobelmag.pgm"
    high_output_file = "C:/Users/leema/Downloads/HW1/HW1/sobelout2.pgm"
    mag_output_file  = "C:/Users/leema/Downloads/HW1/HW1/sobelout1.pgm"

    low_threshold  = 25   # Low threshold value
    high_threshold = 100  # High threshold value

    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        exit(1)

    # Run the Sobel edge detection and save outputs
    sobel_edge_detection(
        input_file, 
        low_output_file, 
        high_output_file, 
        mag_output_file,
        low_threshold, 
        high_threshold
    )

    print(f"Processing complete.")
    print(f"Low threshold result saved to {low_output_file}.")
    print(f"High threshold result saved to {high_output_file}.")
    print(f"Magnitude image saved to {mag_output_file}.")
