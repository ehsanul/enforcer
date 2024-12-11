from sys import argv
import time
import cv2
import os

import numpy as np
from ultralytics import YOLO

def load_session_frames(capture_dir):
    frames = []
    
    # Check if directory exists
    if not os.path.exists(capture_dir):
        raise FileNotFoundError(f"Session directory {capture_dir} not found")
    
    # Get all files and sort them
    files = sorted([f for f in os.listdir(capture_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Load each image file as a cv2 frame
    for file in files:
        file_path = os.path.join(capture_dir, file)
        frame = cv2.imread(file_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not load {file_path}")
    
    return frames

capture_dir = argv[1]
frames = load_session_frames(capture_dir)
model = YOLO("yolo11n.pt")

moving_avg_tv_box = None #np.array([0, 0, 0, 0])

for frame in frames:
    #time.sleep(0.2)
    #cv2.imshow("frame", frame)

    # Run YOLO inference on the frame
    results = model(frame)

    tv_boxes = []

    # Visualize the results on the frame
    for r in results:
        annotated_frame = r.plot()

        # Display the annotated frame
        cv2.imshow("window", annotated_frame)
        time.sleep(0.8)

        """
        for mask in r.masks:
            # Display the annotated frame
            time.sleep(0.2)

            redImg = np.zeros(frame.shape, frame.dtype)
            redImg[:,:] = (0, 0, 255)
            print("===========================================")
            print(type(mask.numpy().data))
            print("===========================================")
            redMask = cv2.bitwise_and(redImg, redImg, mask=mask.numpy().data)
            cv2.addWeighted(redMask, 1, frame, 1, 0, frame)

            cv2.imshow("YOLO mask", frame)
        """

        img = np.copy(r.orig_img)
        # Iterate each detection
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            if label != "tv":
                continue

            for box in c.boxes.xyxy.cpu().numpy():
                tv_boxes.append(box)
                # Crop object from original image

            # b_mask = np.zeros(img.shape[:2], np.uint8)

            # # Create contour mask 
            # contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            # _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)


            # # Choose one:

            # # OPTION-1: Isolate object with black background
            # mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            # isolated = cv2.bitwise_and(mask3ch, img)

            # # # OPTION-2: Isolate object with transparent background (when saved as PNG)
            # # isolated = np.dstack([img, b_mask])

            # # # OPTIONAL: detection crop (from either OPT1 or OPT2)
            # x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            # iso_crop = isolated[y1:y2, x1:x2]
            # cv2.imshow("isolated", isolated)

    if len(tv_boxes) == 1:
        found_box = tv_boxes[0]
    if len(tv_boxes) > 1:
        # union the boxes, maximize the area
        for box in tv_boxes[1:]:
            found_box = np.array([
                min(found_box[0], box[0]),
                min(found_box[1], box[1]),
                max(found_box[2], box[2]),
                max(found_box[3], box[3])
            ])

    if (moving_avg_tv_box is None and len(tv_boxes) > 0):
        moving_avg_tv_box = found_box
    elif len(tv_boxes) > 0:
        moving_avg_tv_box = 0.8 * moving_avg_tv_box + 0.2 * found_box

    box = moving_avg_tv_box 
    crop_obj = img[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
    cv2.imshow(f"cropped", crop_obj)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()