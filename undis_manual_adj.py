import cv2
import numpy as np
import argparse
import datetime
import os

def undistort(img, k1, k2, k3 = 0):
    h, w = img.shape[:2]
    f = max([h, w])
 
    # カメラ行列
    cam_mat = np.array([[f,  0., w / 2],
                       [0., f,  h / 2],
                       [0., 0., 1    ]])
    # 歪み補正パラメータ
    dist_coef = np.array([k1, k2, 0, 0, k3])

    cam_mat_n, roi_size = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coef, (w, h), 1)
    map = cv2.initUndistortRectifyMap(cam_mat, dist_coef, np.eye(3), cam_mat_n, (w, h), cv2.CV_32FC1)
 
    return cv2.remap(img, map[0], map[1], cv2.INTER_CUBIC)

def nothing(x):
    pass

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-i', '--input_files',
        default="sample/hotel.jpg",
        help='input file (image.png)')
    args = argparser.parse_args()

    src = cv2.imread(args.input_files)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = "out"
    os.makedirs(out_folder, exist_ok=True)
    out_filename = now_str

    val1 = 0.5
    val2 = 28
    k1 = 10.0
    k2 = 10.0
    k3 = 10.0

    cv2.namedWindow('track_bar')
    cv2.createTrackbar('val1','track_bar', int(val1*10) , 50, nothing)
    cv2.createTrackbar('val2','track_bar', val2 , 255, nothing)
    cv2.createTrackbar('k1','track_bar', int(k1*10), 200, nothing)
    cv2.createTrackbar('k2','track_bar', int(k2*10), 200, nothing)
    cv2.createTrackbar('k3','track_bar', int(k2*10), 200, nothing)

    while True:
        val1 = cv2.getTrackbarPos('val1','track_bar')
        val2 = cv2.getTrackbarPos('val2','track_bar')
        k1 = cv2.getTrackbarPos('k1','track_bar') / 10 - 10
        k2 = cv2.getTrackbarPos('k2','track_bar') / 10 - 10
        k3 = cv2.getTrackbarPos('k3','track_bar') / 10 - 10

        dst = undistort(src, k1, k2, k3)

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k  & 0xFF == ord('c'):
            cv2.imwrite(f"{out_folder}/{out_filename}.png", dst)

        cv2.putText(dst, f'k1:{k1:.4} k2:{k2:.4} k3:{k3:.4}', (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
        cv2.imshow('dst', dst)


if __name__ == '__main__':
    main()