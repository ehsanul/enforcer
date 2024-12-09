import os
import cv2

# read username and password from dotenv file
from dotenv import load_dotenv # type: ignore
load_dotenv()

username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

cap = cv2.VideoCapture(f'rtsp://{username}:{password}@10.0.0.176:554/Preview_01_main')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()