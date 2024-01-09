import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



# if __name__ == '__main__':
# Get the current directory
current_directory = os.getcwd()
print(current_directory)

# Go back to the parent directory
parent_directory = os.path.dirname(current_directory)
print(parent_directory)

# Set input directory
images_directory   = os.path.join(parent_directory, 'Optical-Flow-Obstacle-Avoidance-for-UAV-main', 'test_img')

# Get a list of image file names in the directory
image_files = [f for f in os.listdir(images_directory ) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(images_directory , image_file)
    original_image = cv2.imread(image_path)
    img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    
    # Check if the image was loaded successfully
    if img is None:
        print("Failed to load image:", image_path)
        exit(1)

    # Reshape the image array
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized) 

    # Define the stopping criteria and number of clusters (K) for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 2
    attempts=10
    # Apply k-means algorithm
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Convert the center values to uint8 and reconstruct the image
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    output_path = '../Optical-Flow-Obstacle-Avoidance-for-UAV-main/output_path'
    plt.savefig(output_path)
    plt.show()
