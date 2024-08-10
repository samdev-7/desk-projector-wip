from distutils import dist
import time
import cv2
import os
import math
import pyapriltags
# import apriltag
import numpy as np 
import calibrate_cam
import calibrate_screen_pos
# import tkinter as tk
# from PIL import Image, ImageTk
# os.environ['DISPLAY'] = '192.168.0.32:10.0'
os.environ['DISPLAY'] = ':0'

detector = pyapriltags.Detector(nthreads=1, families='tag16h5')
# options = apriltag.DetectorOptions(families="tag16h5")
# detector = apriltag.Detector(options)

# app = tk.Tk()
# app.attributes('-fullscreen', True)
# app.config(cursor='none')

# label = tk.Label(app)
# label.pack()

# app.update()
# app.update_idletasks()

# os.environ['DISPLAY'] = '192.168.0.32:10.0'

# testimg = cv2.imread("test.png")

camera_matrix, dist_coeffs = calibrate_cam.load_coefficients()
screen_corners = calibrate_screen_pos.get_screen_pos()
# convert screen_corners to list of tuples of two int
screen_corners = [tuple(map(int, corner[0])) for corner in screen_corners]
lowest = min(screen_corners, key = lambda x: (x[1], x[0]))

screen_corners = sorted(screen_corners, key=lambda x: math.atan2(x[1]-lowest[1], x[0]-lowest[0]) + 2 * math.pi)

rotate = 1

for i in range(rotate):
    screen_corners = screen_corners[1:] + [screen_corners[0]]

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

prespective_transform = cv2.getPerspectiveTransform(
    np.array(screen_corners, dtype=np.float32),
    np.array([
        [0, 0],
        [SCREEN_WIDTH-1, 0],
        [SCREEN_WIDTH-1, SCREEN_HEIGHT-1],
        [0, SCREEN_HEIGHT-1]
    ], dtype=np.float32)
)

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (1280, 960), 1, (1280, 960)
)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (1280, 960), 5) # type: ignore


cv2.namedWindow("image", flags=(cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO))
cv2.setWindowProperty("image", cv2.WND_PROP_TOPMOST, 1.0)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, 1.0)
vc = cv2.VideoCapture("/dev/deskcam")

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
vc.set(cv2.CAP_PROP_FPS, 30)
vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # type: ignore

TIMINGS = False

# newcameramatrix, _ = cv2.getOptimalCameraMatrix(
#     camera_matrix, dist_coeffs, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT), 1, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
# )

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

os.system("~/cam-conf.sh")

last_frame = time.time()

while rval:
    if TIMINGS: captureStart = time.time()
    rval, frame = vc.read()
    if TIMINGS:  print(f"capture time: {time.time() - captureStart}")

    if TIMINGS: undistortStart = time.time()
    # undistorted_image = cv2.undistort(
    #     frame, camera_matrix, dist_coeffs, None, camera_matrix
    # )
    undistorted_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    if TIMINGS: print(f"undistort time: {time.time() - undistortStart}")

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
    if TIMINGS: warpStart = time.time()
    dst = cv2.warpPerspective(undistorted_image, prespective_transform, (SCREEN_WIDTH, SCREEN_HEIGHT))
    # cv2.imshow("screen", dst)
    if TIMINGS: print(f"warp time: {time.time() - warpStart}")

    gimg = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gimg)

    blank = np.zeros_like(dst)

    detectStart = time.time()
    tags = detector.detect(gimg)
    if TIMINGS: print(f"detect time: {time.time() - detectStart}")

    if TIMINGS: cvDrawStart = time.time()
    for tag in tags:
        if tag.hamming > 0:
            continue
        np.random.seed(tag.tag_id)
        color = np.random.randint(0, 255, 3)
        center = tag.center
        new_corners = tag.corners.copy()
        for i, corner in enumerate(tag.corners):
            direction = corner - center
            direction /= np.linalg.norm(direction)
            new_corners[i] += 30 * direction
            corner += 10 * direction
        cv2.fillPoly(blank, [new_corners.astype(np.int32)], color.tolist())
        bottom_center = (new_corners[0] + new_corners[1]) / 2
        cv2.line(blank, tuple(center.astype(np.int32)), tuple(bottom_center.astype(np.int32)), (255, 255, 255), thickness=10)
        cv2.fillPoly(blank, [tag.corners.astype(np.int32)], (0, 0, 0))

        text = ["Hello!", "Testing!", "Apriltags!"][tag.tag_id % 3]
        
        # add testimage
        # target_width = 10
        # target_height = 10
        # x_scale = target_width / testimg.shape[1]
        # y_scale = target_height / testimg.shape[0]
        # testimg_center = (testimg.shape[1]//2, testimg.shape[0]//2)
        # scaled_testimg = cv2.resize(testimg, testimg_center, fx=x_scale, fy=y_scale)
        # x_offset = scaled_testimg.shape[1]//2-130
        # y_offset = scaled_testimg.shape[0]//2-50
        # angle = math.atan2(tag.corners[1][1] - tag.corners[0][1], tag.corners[1][0] - tag.corners[0][0])
        # img_center = (scaled_testimg.shape[1]//2, scaled_testimg.shape[0]//2)
        # # translate the offsets to match the center of the tag and rotation
        # rot_x_offset = int(x_offset * math.cos(angle) - y_offset * math.sin(angle) + center[0])
        # rot_y_offset = int(x_offset * math.sin(angle) + y_offset * math.cos(angle) + center[1])
        # rot_testimg = cv2.warpAffine(scaled_testimg, cv2.getRotationMatrix2D(img_center, -math.degrees(angle), 1), (scaled_testimg.shape[1], scaled_testimg.shape[0]))
        # try:
        #     blank[rot_y_offset:rot_y_offset+rot_testimg.shape[0], rot_x_offset:rot_x_offset+rot_testimg.shape[1]] = cv2.addWeighted(blank[rot_y_offset:rot_y_offset+rot_testimg.shape[0], rot_x_offset:rot_x_offset+rot_testimg.shape[1]], 1, rot_testimg, 1, 0)
        # except:
        #     pass

    fps = 1 / (time.time() - last_frame)
    last_frame = time.time()
    cv2.putText(blank, f"{fps} FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow("image", blank)
    if TIMINGS: print(f"cv draw time: {time.time() - cvDrawStart}")
    # tkDrawStart = time.time()
    # imgarr = Image.fromarray(blank)
    # imgtk = ImageTk.PhotoImage(image=imgarr)
    # label.imgtk = imgtk
    # label.configure(image=imgtk)
    # print(f"tk draw time: {time.time() - tkDrawStart}")

    # updateTime = time.time()
    # app.attributes('-fullscreen', True)
    # app.update()
    # print(f"update time: {time.time() - updateTime}")

    key = cv2.waitKey(1)
    if key == ord('q'): # exit on q
       break
    if TIMINGS: print(f"total time: {time.time() - captureStart}")

# cv2.destroyWindow("preview")
vc.release()
