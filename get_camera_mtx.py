import cv2
import glob
import natsort
import numpy as np

CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objPoints = []
imgPoints = []
objP = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objP[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


if __name__ == "__main__":
    imgList = glob.glob("./samples/hubitec/*.jpg")
    imgList = natsort.natsorted(imgList)
    for imgPath in imgList:
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (960, 540))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH 
                                                + cv2.CALIB_CB_FAST_CHECK 
                                                + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(ret)
        if ret == True:
            objPoints.append(objP)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgPoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("img", img)
        if cv2.waitKey(10) == ord('q'):
            break
    cv2.destroyAllWindows()
    h, w = img.shape[:2]
    print(img.shape[:2])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    for imgPath in imgList[:5]:
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (960, 540))
        newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist)
        dx, dy, dw, dh = roi
        dst2 = cv2.undistort(img, mtx, dist, None, newcameraMtx)
        dst2 = dst2[dy:dy+dh, dx:dx+dw]
        cv2.imshow("1", dst)
        cv2.imshow("2", dst2)     
        print(roi)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()

    print("Camera matrix \n")
    
    print(mtx)

    np.savez("./samples/hubitec/calib.npz", ret = ret, mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs)

    mean_error = 0
    for i in range(len(objPoints)):
        imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i],tvecs[i], mtx, dist);
        error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
        mean_error += error
    print(mean_error/len(objPoints))  
    
    load_mtx = np.load("./samples/hubitec/calib.npz")
    print(load_mtx["mtx"])
    print(load_mtx["dist"])

    # for img in imgs:
    #     src = cv2.imread(img, cv2.IMREAD_COLOR)
    #     cv2.imshow("dd", src)
    #     if cv2.waitKey(0)==ord('q'):
    #         break

