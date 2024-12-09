import os
import cv2
import queue
import threading
import time

# read username and password from dotenv file
from dotenv import load_dotenv
from ksuid import Ksuid # type: ignore
load_dotenv()

username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
camera_ip = os.getenv("CAMERA_IP")

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

# Create VideoCapture instance with RTSP URL
rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:554/Preview_01_main'
cap = VideoCapture(rtsp_url)

session_id = Ksuid()
session_dir = f'captures/{session_id}'
os.makedirs(session_dir, exist_ok=True)

num = 0
while True:
    num += 1
    time.sleep(2)
    frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.imwrite(f'{session_dir}/frame{num:03d}.jpg', frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break

cv2.destroyAllWindows()