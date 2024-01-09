import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the current directory
current_directory = os.getcwd()
print(current_directory)

# Go back to the parent directory
parent_directory = os.path.dirname(current_directory)
print(parent_directory)

# Set input and output directory
video_path = os.path.join(parent_directory, 'Optical-Flow-Obstacle-Avoidance-for-UAV-main', 'test2.mp4')
output_video_path = os.path.join(parent_directory, 'Output', 'test_output.mp4')
print(video_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Shi-Tomasi Parameters
shitomasi_params = dict(maxCorners=100, qualityLevel=0.5, minDistance=7)

# Lucas-Kanade Parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, frame = cap.read()

# Convert the first frame to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get features from Shi-Tomasi
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **shitomasi_params)

# Create an empty mask
mask = np.zeros_like(frame)

# Define obstacle detection parameters
displacement_threshold = 5

# Get the frame dimensions
frame_height, frame_width = frame.shape[:2]

# Calculate the middle coordinates of the frame
middle_x = frame_width // 2
middle_y = frame_height // 2

# Define the size of the ROI
roi_width = 200
roi_height = 200

# Calculate the top-left and bottom-right coordinates of the ROI
roi_x1 = middle_x - (roi_width // 2)
roi_y1 = middle_y - (roi_height // 2)
roi_x2 = roi_x1 + roi_width
roi_y2 = roi_y1 + roi_height

# Define the region of interest [x1, y1, x2, y2]
region_of_interest = [roi_x1-180, roi_y1-150, roi_x2-40, roi_y2+250]

# Initialize lists to store the average displacements
avg_displacement_x_list = []
avg_displacement_y_list = []

displacement_x_list = []
displacement_y_list = []

frame_number = 0
centroid_x = 0
centroid_y = 0
while True:
    # Read the frame from the video
    ret, frame = cap.read()

    # Break the loop if there are no more frames to read
    if not ret:
        break

    frame_number += 1

    # Print the frame number at the top of the frame
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the region of interest
    # cv2.rectangle(frame, (region_of_interest[0], region_of_interest[1]), (region_of_interest[2], region_of_interest[3]), (0, 0, 0), 2)

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_gray is not None and frame_gray_init is not None:
        # Calculate optical flow using Lucas-Kanade
        new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None, **lk_params)

        # Check if the optical flow calculation was successful
        if new_edges is not None:
            # Store the matched features (status=1 means a match)
            good_old = edges[status == 1]
            good_new = new_edges[status == 1]

            # Obstacle detection
            obstacle_detected = False

            for new, old in zip(good_new, good_old):
                x1, y1 = new.ravel()  # (x1,y1):當前像素
                x2, y2 = old.ravel()  # (x2,y2):前一幀像素
                print(f"Pixel ({x1:.4f}, {y1:.4f}): Optical Flow Vector ({x2 - x1:.4f}, {y2 - y1:.4f})")
                
                displacement_x = x1 - x2
                displacement_y = y1 - y2
                print(f'displacement_x: {displacement_x:.4f}')
                print(f'displacement_y: {displacement_y:.4f}')

                # Check if displacement exceeds threshold
                if abs(displacement_x) > displacement_threshold or abs(displacement_y) > displacement_threshold:
                    # Potential obstacle detected
                    obstacle_detected = True
                    mask = cv2.arrowedLine(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), thickness=-1)

                    # Save the average displacements
                    displacement_x_list.append(displacement_x)
                    displacement_y_list.append(displacement_y)

            # Check if we have enough corners to perform analysis | Min = 2
            if len(good_new) > 2:
                # Get the corners that are in the ROI
                roi_corners = good_new[
                    (good_new[:, 0] >= region_of_interest[0]) &
                    (good_new[:, 1] >= region_of_interest[1]) &
                    (good_new[:, 0] <= region_of_interest[2]) &
                    (good_new[:, 1] <= region_of_interest[3])
                ]
                # Check if we have enough corners in the ROI
                if len(roi_corners) > 2:
                    # Calculate the centroid of the corners within the ROI
                    centroid_x = int(np.nanmean(roi_corners[:, 0]))
                    centroid_y = int(np.nanmean(roi_corners[:, 1]))

                    # Output the centroid coordinates
                    print(f"Centroid: ({centroid_x}, {centroid_y})")

            # Draw a bounding box around the centroid
            box_size = 150  # You can adjust the size of the bounding box
            # cv2.rectangle(frame, (centroid_x - box_size, centroid_y - box_size), (centroid_x + box_size, centroid_y + box_size), (255, 0, 0), 2)

            # Overlay the optical flow arrows on the original frame
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

            # Display the frame with obstacle detection and optical flow
            output = cv2.add(frame, mask)
            cv2.imshow('Obstacle Detection and Optical Flow', output)

            # Wait for the 'q' key to be pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Update the previous frame and edges
            frame_gray_init = frame_gray.copy()
            edges = good_new.reshape(-1, 1, 2)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
