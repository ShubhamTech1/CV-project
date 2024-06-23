
# load the image dataset
import cv2
import glob 


# for single image loading

image_path = r"D:\NLP, CV\CV\test1\test1\108.jpg"
image = cv2.imread(image_path)
# Display the image in a separate window
cv2.imshow("Original_Image", image)
# Wait for a key press to close the window (optional, but recommended for interactive viewing)
cv2.waitKey(0)
# Close all windows (optional)
cv2.destroyAllWindows()












                        # GENERAL IMAGE PREPROCESSING TECHNIQUE
                        
# 1] RESIZING AND CROPPING IMAGE 

                                 # RESIZING
# Define new width and height
new_width = 300
new_height = 200
# Resize the image using interpolation (optional)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

#Define scaling factor (0 to 1 for scaling down)
#scale_factor = 0.5
#resized_image1 = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

cv2.imshow("Resized Image", resized_image)  # Display the resized image
cv2.waitKey(0)

cv2.imwrite("resized_image.jpg", resized_image) # Save the resized image (optional)


                                    # CROPPING
                        
# Define top-left corner and bottom-right corner coordinates of the ROI
top_left_x = 100
top_left_y = 50
bottom_right_x = 300
bottom_right_y = 250

# Extract the ROI using slicing
cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]













# 2] IMAGE SCALING

                            # 1] NORMALIZATION 

import numpy as np 
# Find minimum and maximum pixel intensities
min_value = np.min(image)
max_value = np.max(image)
# Normalize the image using min-max scaling (0 to 1)
normalized_image = (image - min_value) / (max_value - min_value)

cv2.imshow("normalized_image", normalized_image) 
cv2.waitKey(0)


                            # 2] STANDARDISATION

# Calculate mean and standard deviation
mean = np.mean(image)
std = np.std(image)

# Normalize the image using Z-score normalization
standardized_image = (image - mean) / std
cv2.imshow("standardized_image", standardized_image) 
cv2.waitKey(0)










# 3] COLOR SPACE CONVERSION

conversion1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert color image into grayscale image -only one channel
cv2.imshow("gray", conversion1) 
cv2.waitKey(0)

conversion2 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)  # convert into saturated color  
cv2.imshow("saturated", conversion2)             
cv2.waitKey(0)


conversion3 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # convert into saturated color  
cv2.imshow("color",conversion3) 
cv2.waitKey(0)















# 4] Noise Reduction/ Denoising
'''
2 Techniques :- 1] filtering Techniques :- Linear_Smoothing:- 1] mean filtering
                                                              2] gaussian filtering

                                           NonLinear_smoothing:- 1] median filtering
                                                                 2] billateralfiltering
# this technique is basically used for noise reduction
                2] statistical Technique :- Wavelet Denoising (advanced filtering technique)

'''
# Linear_Smoothing:- 1] mean filtering

kernel_size = 5  # Experiment for best results (larger kernel for more smoothing)
# Apply mean filtering using cv2.blur (faster for box filter)
image_blur = cv2.blur(image, (kernel_size, kernel_size))
cv2.imshow("mean_filter", image_blur)
cv2.waitKey(0)

#                    2] gaussian filtering
kernel_size = 5
# Apply Gaussian filtering
image_gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
cv2.imshow("gaussian_filter", image_gaussian)
cv2.waitKey(0)



# NonLinear_smoothing:- 1] median filtering
kernel_size = 5
# Apply median filtering
image_median = cv2.medianBlur(image, kernel_size)
cv2.imshow("median_filter", image_median)
cv2.waitKey(0)

#                       2] billateralfiltering

d = 9  # Diameter of neighborhood
sigmaColor = 75  # Color space sigma
sigmaSpace = 75  # Coordinate space sigma

denoised_image_bilateral = cv2.bilateralFiltering(image, d, sigmaColor, sigmaSpace)
# AttributeError: module 'cv2' has no attribute 'bilateralFiltering'
cv2.__version__













# 5] Contrast Enhancement

conversion1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert color image into grayscale 
# now i enhance contrast for this gray image
contrast_enhance = cv2.equalizeHist(conversion1) 
cv2.imshow("gray",conversion1)
cv2.imshow("after", contrast_enhance)
cv2.waitKey(0)














                        # TASK BASED PREPROCESSING TECHNIQUES:-

# TASK:- IMAGE CLASSIFICATION , OBJECT DETECTION AND RECOGNITION
# TECHNIQUE :- 1) DATA AUGMENTATION
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import cv2
import numpy as np

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation angle in degrees
    width_shift_range=0.2,  # Random horizontal shift as a fraction of image width
    height_shift_range=0.2,  # Random vertical shift as a fraction of image height
    shear_range=0.2,  # Random shear angle in degrees
    zoom_range=0.2,  # Random zoom range
    horizontal_flip=True,  # Random horizontal flipping
    fill_mode='nearest'  # Pixel filling mode for padding
)

# Load your image (replace with your loading logic)
image_path = r"D:\NLP, CV\CV\test1\test1\108.jpg"
image = cv2.imread(image_path)
image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch processing

# Augment the image (demonstration with a single image)
augmented_images = datagen.flow(image, batch_size=1)  # Batch size of 1 for single image

# Get the first augmented image from the generator
augmented_image = next(augmented_images)[0]

# Convert back to uint8 for display (if applicable)
augmented_image = cv2.convertScaleAbs(augmented_image, alpha=1.0, beta=0.0)

# Display original and augmented image
cv2.imshow("Original Image", image[0])  # Access the image from the batch dimension
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Data augmentation complete!")


#         2) Morphological Transformation

#1] erosion 
#2] dilation

import numpy as np 
# Define structuring element (kernel) - shape used for erosion/dilation
kernel = np.ones((3, 3), np.uint8)  # Example: 3x3 square kernel

eroded_image = cv2.erode(image, kernel) 

# Dilation
dilated_image = cv2.dilate(image, kernel)

# Display original, eroded, and dilated images
cv2.imshow("Eroded Image", eroded_image)
cv2.imshow("Dilated Image", dilated_image)
cv2.waitKey(0)













#TASK :-IMAGE SEGMENTATION 
#TECHNIQUE:- THRESHOLDING
 
# Convert to grayscale (if necessary)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1] Simple Thresholding (replace threshold value as needed)
threshold = 60  # Experiment with different threshold values
ret, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

# 2] Adaptive Thresholding Example (using Mean C method)
thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)

# Display original, simple thresholded, and adaptive thresholded images
cv2.imshow("Original Image", image)
cv2.imshow("Simple Thresholded", binary_image)
cv2.imshow("Adaptive Thresholded (Mean C)", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()






def deskew_image(image):
  # Convert to grayscale (if necessary)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply skew detection (replace with your preferred method)
  skew_angle = transform.estimate_skew_angle(gray)

  # Rotate the image to correct skew
  corrected_image = transform.rotate(image, -skew_angle, preserve_range=True)

  return corrected_image


def binarize_image(image):
  # Convert to grayscale (if necessary)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Otsu's thresholding (adjust as needed)
  thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return binary_image


def thin_image(binary_image):
  # Implementing thinning algorithm (replace with a robust thinning algorithm)
  thinned_image = np.zeros_like(binary_image)
  # ... (implementation of thinning logic here)  # Replace with a proper thinning algorithm
  return thinned_image


def process_image(image):
  # Load the image
  image = cv2.imread(image)

  # Correct skew (optional)
  # corrected_image = deskew_image(image)

  # Binarize the image
  binary_image = binarize_image(image)

  # Thin the image (replace with a proper thinning algorithm)
  thinned_image = thin_image(binary_image)

  # Display or save the processed images (adjust as needed)
  cv2.imshow("Original Image", image)
  cv2.imshow("Binary Image", binary_image)
  cv2.imshow("Thinned Image", thinned_image)  # Uncomment if thinning is implemented
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Example usage
image_path = "path/to/your/image.jpg"
process_image(image_path)

















'''
import os
# load the folder which contain multiple images

image_folder = r"D:\NLP, CV\CV\test1\test1"
# List to store loaded images
loaded_images = []
# Get all image paths in the folder (adjust for your image format)
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(".jpg")]

for image_path in image_paths:
  # Load the image using cv2.imread()
  image = cv2.imread(image_path)
  # Add the loaded image to the list
  loaded_images.append(image)
      
len(loaded_images)

# Loop through loaded images and display them
for image in loaded_images:
  cv2.imshow("Image", image)
  cv2.waitKey(0)  # Wait for a key press to close the window before showing the next image

# Close all windows (optional)
cv2.destroyAllWindows()

'''



# HISTOGRAM OF RBG CHANNEL IMAGE:- how pixel intensity spread of evey channel (RGB)

import cv2
# Alternatively, use matplotlib.pyplot (plt) for image loading

# Load the image
image = cv2.imread(r"C:\Users\Asus\Documents\Pictures\Screenshots\Screenshot 2024-05-15 191446.png")
# Split the image into its three channels (BGR order in OpenCV)
b, g, r = cv2.split(image)
# Plotting histograms (using matplotlib.pyplot here)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot red channel histogram
plt.hist(r.ravel(), bins=256, range=(0, 255), density=True, alpha=0.5, label='Red Channel')

# Plot green channel histogram (similarly for blue)
plt.hist(g.ravel(), bins=256, range=(0, 255), density=True, alpha=0.5, label='Green Channel')

# Plot blue channel histogram
plt.hist(b.ravel(), bins=256, range=(0, 255), density=True, alpha=0.5, label='Blue Channel')

plt.legend()
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability Density")
plt.title("Histogram of RGB Channels")
plt.grid(True)
plt.show()




