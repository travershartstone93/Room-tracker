import cv2
import numpy as np
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cameras', type=int, default=2)
args = parser.parse_args()

CHESS_SIZE = (7, 6)
objp = np.zeros((CHESS_SIZE[0] * CHESS_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_SIZE[0], 0:CHESS_SIZE[1]].T.reshape(-1, 2)

imgpoints = [[] for _ in range(args.cameras)]
objpoints = []

caps = [cv2.VideoCapture(i) for i in range(args.cameras)]

print("Press 'c' to capture, 'q' to quit.")
while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            cv2.imshow(f'cam{len(frames)-1}', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        rets = []
        corners_list = []
        for gray in grays:
            ret, corners = cv2.findChessboardCorners(gray, CHESS_SIZE)
            rets.append(ret)
            corners_list.append(corners)
        if all(rets):
            objpoints.append(objp)
            for i in range(args.cameras):
                imgpoints[i].append(corners_list[i])
            print("Captured image set.")

for cap in caps:
    cap.release()
cv2.destroyAllWindows()

# Calibrate each camera
calib = {}
for i in range(args.cameras):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints[i], frames[0].shape[1::-1], None, None)
    calib[f'cam{i}'] = {'K': K.tolist(), 'dist': dist.tolist()}

with open('config/calibration.yaml', 'w') as f:
    yaml.dump(calib, f)

print("Calibration saved to config/calibration.yaml")
# Note: Extrinsics not estimated here; use SLAM for that.
