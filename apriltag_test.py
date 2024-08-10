import cv2
import os
import calibrate_cam
import apriltag
import numpy as np

os.environ['DISPLAY'] = '192.168.0.32:10.0'

camera_matrix, dist_coeffs = calibrate_cam.load_coefficients()

options = apriltag.DetectorOptions(families="tag16h5")
detector = apriltag.Detector(options)

cv2.namedWindow("preview")
vc = cv2.VideoCapture("/dev/deskcam")

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# newcameramatrix, _ = cv2.getOptimalCameraMatrix(
#     camera_matrix, dist_coeffs, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT), 1, (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
# )

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    img = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, camera_matrix
    )
else:
    rval = False

os.system("~/cam-conf.sh")

while rval:
    cv2.imshow("preview", img)
    rval, frame = vc.read()
    img = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, camera_matrix
    )

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gimg)

    for det in detections:
        for idx in range(len(det.corners)):
            pt1 = det.corners[idx]
            pt2 = det.corners[(idx + 1) % len(det.corners)]
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0))
        cv2.putText(img, str(det.tag_id), (int(det.center[0]), int(det.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    key = cv2.waitKey(20)
    if key == ord('q'): # exit on q
       break

cv2.destroyWindow("preview")
vc.release()
