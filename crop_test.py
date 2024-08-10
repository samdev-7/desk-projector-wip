import cv2
import os
import math
import numpy as np 
import calibrate_cam
import calibrate_screen_pos
import tkinter as tk
from PIL import Image, ImageTk

os.environ['DISPLAY'] = ':0'

app = tk.Tk()
app.attributes('-fullscreen', True)
app.config(cursor='none')

label = tk.Label(app)
label.pack()

# os.environ['DISPLAY'] = '192.168.0.32:10.0'

camera_matrix, dist_coeffs = calibrate_cam.load_coefficients()
screen_corners = calibrate_screen_pos.get_screen_pos()
# convert screen_corners to list of tuples of two int
screen_corners = [tuple(map(int, corner[0])) for corner in screen_corners]
lowest = min(screen_corners, key = lambda x: (x[1], x[0]))

screen_corners = sorted(screen_corners, key=lambda x: math.atan2(x[1]-lowest[1], x[0]-lowest[0]) + 2 * math.pi)

rotate = 1

for i in range(rotate):
    screen_corners = screen_corners[1:] + [screen_corners[0]]

# cv2.namedWindow("preview")
vc = cv2.VideoCapture("/dev/deskcam")

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# vc.set(cv2.CAP_PROP_FPS, 30)
# vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # type: ignore

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# newcameramatrix, _ = cv2.getOptimalCameraMatrix(
#     camera_matrix, dist_coeffs, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT), 1, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
# )
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (1280, 960), 1, (1280, 960)
)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (1280, 960), 5)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

os.system("~/cam-conf.sh")

while rval:
    rval, frame = vc.read()
    # undistorted_image = cv2.undistort(
    #     frame, camera_matrix, dist_coeffs, None, camera_matrix
    # )
    undistorted_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # label screen corners 1-4
    # for i, corner in enumerate(screen_corners):
    #     cv2.circle(undistorted_image, corner, 5, (0, 0, 255), -1)
    #     cv2.putText(undistorted_image, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # # draw lines between corners
    # for i in range(4):
    #     cv2.line(undistorted_image, screen_corners[i], screen_corners[(i+1)%4], (0, 0, 255), 2)
    # cv2.imshow("preview", undistorted_image)

    # create a new image and stretch the image based on the screen corners to fill it 
    # (this is the image that will be displayed on the screen)
    dst = cv2.warpPerspective(undistorted_image, cv2.getPerspectiveTransform(
        np.array(screen_corners, dtype=np.float32),
        np.array([
            [0, 0],
            [SCREEN_WIDTH-1, 0],
            [SCREEN_WIDTH-1, SCREEN_HEIGHT-1],
            [0, SCREEN_HEIGHT-1]
        ], dtype=np.float32)
    ), (SCREEN_WIDTH, SCREEN_HEIGHT))
    # cv2.imshow("screen", dst)

    gimg = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gimg, 128, 255, cv2.THRESH_BINARY)

    imgarr = Image.fromarray(dst)
    imgtk = ImageTk.PhotoImage(image=imgarr)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    app.update()

    key = cv2.waitKey(10)
    if key == ord('q'): # exit on q
       break

# cv2.destroyWindow("preview")
vc.release()
