import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

video_path = 'D:/image_experience/Optical-Flow-Obstacle-Avoidance-for-UAV-main/test2.mp4'
print(video_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Farneback Parameters
farneback_params = dict(pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)

# Read the first frame
ret, frame = cap.read()

# Convert the first frame to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define obstacle detection parameters
displacement_threshold = 5

# Get the frame dimensions
frame_height, frame_width = frame.shape[:2]

# Calculate the middle coordinates of the frame
middle_x = frame_width // 2
middle_y = frame_height // 2

# # Define the size of the ROI
# roi_width = 200
# roi_height = 200

# # Calculate the top-left and bottom-right coordinates of the ROI
# roi_x1 = middle_x - (roi_width // 2)
# roi_y1 = middle_y - (roi_height // 2)
# roi_x2 = roi_x1 + roi_width
# roi_y2 = roi_y1 + roi_height

# # Define the region of interest [x1, y1, x2, y2]
# region_of_interest = [roi_x1-180, roi_y1-150, roi_x2-40, roi_y2+250]

# Initialize lists to store the average displacements
avg_displacement_x_list = []
avg_displacement_y_list = []

frame_number = 0
frame_count = 0
start_time = time.time()

while True:
    # Read the frame from the video
    ret, frame = cap.read()

    # Break the loop if there are no more frames to read
    if not ret:
        break

    frame_number += 1
    frame_count += 1
    # 顯示當前幀數
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= 1.0:
        frames_per_second = frame_count / elapsed_time
        print(f"Frames per second: {frames_per_second:.2f}")
        # 重置計數器和開始時間
        frame_count = 0
        start_time = time.time()

    # Draw the region of interest
    # cv2.rectangle(frame, (region_of_interest[0], region_of_interest[1]), (region_of_interest[2], region_of_interest[3]), (0, 0, 0), 2)

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_gray is not None and frame_gray_init is not None:
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev=frame_gray_init, next=frame_gray, flow=None, **farneback_params)

        # Calculate the magnitude and angle of the optical flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        gamma = 0.5
        epsilon = 1e-8
        # Normalize the magnitude to 0-255 range
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_normalized = np.power((magnitude_normalized+ epsilon) / 255.0, gamma) * 255
        magnitude_normalized = np.nan_to_num(magnitude_normalized)
        
        # Convert the normalized magnitude to a color map
        flow_color = cv2.applyColorMap(magnitude_normalized.astype(np.uint8), cv2.COLORMAP_RAINBOW)

        
        flow_gray = cv2.cvtColor(flow_color, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(flow_gray, 128, 170, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_image = flow_gray.copy()
        for contour in contours:
            # 忽略太小的輪廓
            # if cv2.contourArea(contour) > 1000:
                # 繪製矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                margin = 300  #設定邊緣寬度(過濾邊緣的矩形)
                if (x > margin) and (y > margin) and (x + w) < (frame_gray.shape[1] - margin) and (y + h) < (frame_gray.shape[0] - margin):
                    # 繪製輪廓
                    cv2.drawContours(result_image, [contour], 0, (255, 255, 255), 2)

                    cv2.rectangle(frame_gray_init, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.rectangle(flow_color, (x, y), (x + w, y + h), (255, 255, 255), 2)     

                    # for i in range(y, y + h):
                    #     for j in range(x, x + w):
                    #         b, g, r = frame[i, j]
                    #         print(f"Pixel ({i}, {j}): B={b}, G={g}, R={r}")
        #---------------------------outputs---------------------------
        resize_factor = 0.3  # Adjust the resize factor as needed
        frame_gray_resized = cv2.resize(frame_gray_init, None, fx=resize_factor, fy=resize_factor)
        flow_color_resized = cv2.resize(flow_color, None, fx=resize_factor, fy=resize_factor)
        result_image_resized = cv2.resize(result_image, None, fx=resize_factor, fy=resize_factor)
        
        # Print the frame number at the top of the frame
        cv2.putText(frame_gray_resized, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(flow_color_resized, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result_image_resized, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Original image', frame_gray_resized)
        cv2.imshow('Colorized Optical Flow', flow_color_resized)
        cv2.imshow('Contours and Rectangles', result_image_resized)

        
        # Wait for the 'q' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame
        frame_gray_init = frame_gray.copy()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
