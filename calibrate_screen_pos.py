import os
import cv2
import time
import tkinter as tk
import numpy as np
import calibrate_cam

def calibrate():
    print("Fetching camera matrix and distortion coefficients")
    camera_matrix, dist_coeffs = calibrate_cam.load_coefficients()

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (1280, 960), 1, (1280, 960)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (1280, 960), 5)

    print("Intializing Debug GUI")
    os.environ['DISPLAY'] = '192.168.0.32:10.0'
    cv2.namedWindow("Debug")

    print("Initializing Main GUI")
    os.environ['DISPLAY'] = ':0'
    app = tk.Tk()

    print("Loading calibration image")
    img = tk.PhotoImage(file='calwhite.png')
    label = tk.Label(app, image=img)
    label.pack()
    app.update()
    app.update_idletasks()

    app.attributes('-fullscreen', True)
    app.config(cursor='none')

    print("Updating GUI with calibration image", end="")
    for _ in range(10):
        app.update()
        app.update_idletasks()
        print(".", end="")
        time.sleep(0.1)
    print()

    print("Initializing camera", end="")
    vc = cv2.VideoCapture("/dev/deskcam")
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    if not vc.isOpened():
        print("Failed to initialize camera")
        exit(1)
    rval, frame = vc.read()
    os.system("~/cam-conf.sh")
    os.system("v4l2-ctl -d /dev/deskcam -c exposure_time_absolute=30")
    print("Waiting 10 seconds for camera to adjust", end="")
    for _ in range(10):
        time.sleep(1)
        print(".", end="")
    print()

    frames = []
    for i in range(10):
        print(f"Capturing frame {i}")
        rval, frame = vc.read()

        print("Undistorting frame")
        # uimg = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
        uimg = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


        print("Converting frame to grayscale")
        gimg = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
        frames.append(gimg)
        last_frame = gimg

    avg = np.mean(np.stack(frames, axis=-1), axis=-1).astype(np.uint8)
    avg = cv2.resize(avg, (last_frame.shape[1], last_frame.shape[0]))

    blur = cv2.GaussianBlur(avg, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)

    kernel = np.ones((5,5), np.uint8)
    rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    mask = np.zeros_like(avg)
    cv2.drawContours(mask, [contours[0]], -1, (255,255,255), -1)

    corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.1, minDistance=150)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(avg,(int(x),int(y)),1,(255,120,255),-1)
        print("({}, {})".format(x,y))

    cv2.imshow("thresh", thresh)
    cv2.imshow("rect", rect)
    cv2.imshow("mask", mask)
    # cv2.imshow("dial", dial)
    # cv2.imshow("diff", diff)
    cv2.imshow("corners", avg)

    print("Saving screen positions to screen_pos.yaml")
    cv_file = cv2.FileStorage("screen_pos.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("corners", corners)
    cv_file.release()

    print("Reset camera settings")
    os.system("~/cam-conf.sh")

def get_screen_pos():
    cv_file = cv2.FileStorage("screen_pos.yaml", cv2.FILE_STORAGE_READ)
    corners = cv_file.getNode("corners").mat()
    cv_file.release()
    return corners

if __name__ == '__main__':
    calibrate()
    print("Calibration is finished")
    print("Waiting for 'q' to exit")
    while True:
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    exit(0)