import numpy as np
import cv2
import math

def make_map_resize(src_w, src_h, scale=2.0):
    dst_w = int(src_w * scale)
    dst_h = int(src_h * scale)
    map = np.zeros((dst_h, dst_w, 2), np.float32)
    for y in range(dst_h):
        for x in range(dst_w):
            map[y][x][0] = x / scale
            map[y][x][1] = y / scale

    return map[:,:,0], map[:,:,1]

def make_map_inv(map_src):
    h, w = map_src[0].shape
    map_nop = np.zeros((h, w, 2), np.float32)
    for y in range(h):
        for x in range(w):
            map_nop[y][x][0] = x
            map_nop[y][x][1] = y

    map = cv2.remap(map_nop, map_src[0], map_src[1], cv2.INTER_NEAREST)

    map_inv = np.zeros((h, w, 2), np.float32)
    for y in range(h):
        for x in range(w):
            fx, ix = math.modf(map[y][x][0])
            fy, iy = math.modf(map[y][x][1])          
            map_inv[int(iy)][int(ix)][0] =  x
            map_inv[int(iy)][int(ix)][1] =  y
 
    return map_inv[:,:,0], map_inv[:,:,1]


def test():
    src = cv2.imread("sample/hotel.jpg")
    src = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2))
    h, w, _ = src.shape
    f = max([w, h])
    cv2.imshow("src", src)

    cam_mat = np.array([[f,     0., w/2],
                        [0.,    f,  h/2],
                        [0.,    0., 1. ]])
    dist_coef = np.array([3.0, 4.0, 1.0, 0, 0])

    new_cam_mat, roi_size = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coef, (w, h), 1)
    map_undis = cv2.initUndistortRectifyMap(cam_mat, dist_coef, None, new_cam_mat, (w, h), cv2.CV_32FC1)
    undist = cv2.remap(src, map_undis[0], map_undis[1], cv2.INTER_CUBIC)
    cv2.imshow("undis", undist)

    map_inv = make_map_inv(map_undis)
    redist = cv2.remap(undist, map_inv[0], map_inv[1], cv2.INTER_CUBIC)
    cv2.imshow("redist", redist)

    cv2.waitKey(0)


if __name__ == '__main__':
    test()