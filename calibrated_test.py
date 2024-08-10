import cv2
import os
import calibrate_cam

os.environ['DISPLAY'] = '192.168.0.32:10.0'

camera_matrix, dist_coeffs = calibrate_cam.load_coefficients()

cv2.namedWindow("preview")
vc = cv2.VideoCapture("/dev/video1")

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# newcameramatrix, _ = cv2.getOptimalCameraMatrix(
#     camera_matrix, dist_coeffs, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT), 1, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
# )

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    undistorted_image = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, camera_matrix
    )
else:
    rval = False

os.system("~/cam-conf.sh")

while rval:
    cv2.imshow("preview", undistorted_image)
    rval, frame = vc.read()
    undistorted_image = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, camera_matrix
    )
    key = cv2.waitKey(1)
    if key == ord('q'): # exit on q
       break

cv2.destroyWindow("preview")
vc.release()
