# prepare object points
nx = 7 #number of inside corners in x
ny = 5 #number of inside corners in y
import time
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt








# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
objp1 = np.zeros((ny*nx, 3), np.float32)
objp1[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
objpoints1 = [] # 3d point in real world space for last object
imgpoints = [] # 2d points in image plane.
imgpoints1 = [] # 2d points in image plane for last object
#get images
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(9,9),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        plt.ion()
        img = cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
        #show every corner in every image
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.1)
    else:
        print(fname) #shows not executable images
# print("image points \n", imgpoints)
# print("object points \n", objpoints)
#Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#dist – Output vector of distortion coefficients
# (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements.
#rvecs – Output vector of rotation vectors
#ret-
#mtx- Output 3x3 floating-point camera matrix
#tvecs – Output vector of translation vectors estimated for each pattern view.
#Store the camera matrix and distortion coefficients in the folder
print('Camera matrix and distortion coefficients were stored in folder Results')
np.savez('Results',ret=ret,matrix=mtx,distance=dist,rotation=rvecs,translation=tvecs)

# Read in an image
img = cv2.imread('1.jpg')
h,  w = img.shape[:2]
#returns the new(optimal)camera matrix based on the free scaling parameter
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibration_result.png',dst)

#Plotting
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.suptitle('PART 1: CALIBRATION OF THE CAMERA')
plt.show()
plt.pause(6)
plt.close()
#calculating the reprojection error
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print("reprojection error = %.2f \n" % (tot_error/len(objpoints)))

#PART2: CALCULATING OBJECT SIZE
#Read image with object
print('PART2:CALCULATING OBJECT SIZE')
imag = cv2.imread('object.jpg')
hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
dst = cv2.undistort(imag, mtx, dist, None, newcameramtx)
#crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('res_object.png',dst)
#DRAW A BOUNDING BOX
im = cv2.imread('res_object.png')
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# Find the chess board corners for that photo
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, add object points, image points (after refining them)
if ret == True:
        objpoints1.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(9,9),(-1,-1),criteria)
        imgpoints1.append(corners2)
        # Draw and display the corners
        # plt.ion()
        img = cv2.drawChessboardCorners(im, (nx, ny), corners2, ret)
        # plt.imshow(im)
        # plt.show()
        # plt.pause(0.5)
else:
    print(fname) #shows not executable images
#print("image points for current photo \n", imgpoints1)
#Range of out color(mandarin) to segment it for next detecting object
COLOR_MIN = np.array([0, 125, 100],np.uint8)
COLOR_MAX = np.array([35, 255, 255],np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
imgray = frame_threshed
ret,thresh = cv2.threshold(frame_threshed,127,255,0)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]
#Obtaining parameters of object
x,y,w1,h1 = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w1,y+h1),(0,255,0),2)
#Plotting
plt.imshow(im, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.suptitle('STEP2: ESTIMATING THE OBJECT SIZE')
plt.title('Object found')
plt.show()
plt.pause(6)
plt.close()

#calculate distance between corners for object
x1 = imgpoints1[0][0][0][0]
y1 = imgpoints1[0][0][0][1]
x2 = imgpoints1[0][1][0][0]
y2 = imgpoints1[0][1][0][1]
print('Points of two corners in the chessboard on our photo:',x1,y1,x2,y2)
a = np.array(x1,y1)
b = np.array(x2,y2)
#calculating Euclidian distance between corners' points
dist1 = np.linalg.norm(a-b)
print('Distance between two pixel points=',dist1)
obj_width = 33 # real size of chessboard lines
ratio = dist1/obj_width #scaling factor
print('Ratio=',ratio)
# plt.imshow(imag)
# plt.show()
#calculate object size with manually estimated distance(use imag = cv2.imread('object.jpg') in string 104)
WD = 70 #distance to object in cm(manually estimated)
#object size in image plane(in pixels)
x_pix = w1
y_pix = h1
print('Object size in pixels=',x_pix,'\n',y_pix)
focal = (mtx[0][0]+mtx[1][1])/(2*ratio) #real focal length,we take into account that fx almost equal to fy
real_size_x = (WD*x_pix/ratio)/focal
real_size_y = (WD*y_pix/ratio)/focal
print('X real size(mm)=',real_size_x,'\nY real size(mm)=',real_size_y)
#How we can see, estimated size is approximetly equal to real size of object
# print(mtx[0][0])
# print(mtx[1][1])


#Step3(a):ESTIMATING THE DISTANCE BETWEEN THE OBJECT AND THE CAMERA(MOUSE)
print('\nPART3(First method):ESTIMATING THE DISTANCE BETWEEN THE OBJECT AND CAMERA\n')
distance_to_obj = (real_size_x*mtx[0][0])/(x_pix)
print('Estimated distance to object (cm) =',distance_to_obj)
#to calculate distance to object2.jpg(distance from camera = 106) or calculate distance to object3_94
# (distance from camera = 94) you need to go to string 105 and instead of
#imag = cv2.imread('object.jpg') write imag = cv2.imread('object2.jpg') or imag = cv2.imread('object3_94.jpg')
# and then in string 192 instead of
#distance_to_obj = (real_size_x*mtx[0][0])/(x_pix) write distance_to_obj = (6.5*mtx[0][0])/(x_pix)
#where 6.5 (cm) - real size of object by x axis.
#We take into accounting that varies only pixel parameters of object(x_pix,y_pix) and focal length, that we can
#obtain from mtx - mtx[0][0]
print('\nPART3(Second method):ESTIMATING THE DISTANCE BETWEEN THE OBJECT AND CAMERA\n')
#OR YOU CAN YOU THE CODE BELOW:
width = 65 #(real size of object)
im = cv2.imread('object2.jpg')
h, w = im.shape[:2]
#Undistort the image
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#undistort
dst = cv2.undistort(im, mtx, dist, None, newcameramtx)
cv2.imwrite('object_distance_calibrated.png', dst)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
im = dst
#Finding the Mouse
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
COLOR_MIN = np.array([0, 125, 100], np.uint8)
COLOR_MAX = np.array([35, 255, 255], np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
imgray = frame_threshed
ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(im, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.suptitle('STEP3: ESTIMATING THE DISTANCE BETWEEN THE OBJECT AND THE CAMERA')
plt.title('Distance found')
plt.show()
plt.pause(6)
print(" The distance between the image plane and the object = %.2f mm" % (width*mtx[0][0]/w))