
from cv2 import (circle, LINE_AA, putText, FONT_HERSHEY_SIMPLEX, projectPoints, cvtColor, imread, COLOR_BGR2GRAY,
                 polylines, resize, solvePnP, Rodrigues, imshow, waitKey, destroyAllWindows, SOLVEPNP_ITERATIVE)
import numpy as np
import scipy.linalg as la
from pupil_apriltags import Detector
from math import sqrt, acos
from support.io.lidar_truth import TruthPoints

tag_size=0.168
scale = 2848.0 / 1424
aprilImage = "C:\\repos\\aburn\\usr\\24WintCalspanFltTest\\AlviumLJAprilTags\\1.bmp"

paramsOnCalibration = True
paramsOnDistortion = True
applyDistortion = True

sss = np.zeros((3,3,3)) # Create basis function for skew symmetric matrices
sss[0,1,2] = 1
sss[0,2,1] = -1
sss[1,0,2] = -1
sss[1,2,0] = 1
sss[2,0,1] = 1
sss[2,1,0] = -1

def v2SS(v):
    '''
    Take in a 3 vector and convert it to a skew-symmetric matrix
    '''
    return np.tensordot(v,sss,axes=([0],[0]))
def v2DCM(v):
    '''
    Take in a 3 vector and use the matrix exponential to create
    a DCM
    '''
    return la.expm(v2SS(v))

def DCM2v(C):
    '''
    Take in a 3x3 DCM and convert it into a Rodrigues vector
    (axis angle where the axis is scaled by the angle of rotation)
    '''
    trace_C = np.trace(C)
    # Rather than explicitly pull out elements of C, I use sss & sum to get out the elements I want
    # This makes the code more portable in case I change sss later on.  :)
    off_diags = np.array([np.sum(SS*C) for SS in sss ])
    if trace_C > 2.999995: # assume theta/sin_theta = 1
        return off_diags/2
    if trace_C < -.999999:
        # First, need to determine the magnitude of each element of the vector...
        S = C + C.T + (1-np.trace(C))*np.eye(3)
        if (3-np.trace(C)) <=0.000001:
            mag_vals = np.sqrt((np.diag(S)/(3-np.trace(C))))
        else:
            mag_vals = np.sqrt((np.diag(np.abs(S))/0.00001))
        # Second, need to figure out the sign for each of the mag_vals
        # Start with getting the relative signs
        main_ax = np.argmax(mag_vals)
        for i in range(3):
            if S[main_ax,i]<0:
                mag_vals[i]  = -mag_vals[i]
        # The axis is now fixed up to a universal sign.  Figure out the universal sign
        if off_diags[main_ax]<0:
            mag_vals = -mag_vals
        # This part is pretty normal ... how big is theta?
        if np.trace(C) < -1:
            theta = acos(-1.0)
        else:
            theta = acos((np.trace(C)-1)/2)
        return theta * mag_vals
    sin_theta = sqrt((3-trace_C)*(1+trace_C))/2.0
    theta = acos((trace_C-1)/2.0)
    return theta/(2. * sin_theta) * off_diags

def plotOnImg(img, points, names, color):
    for idx, pxPt in enumerate(points):
        circle(img, (int(pxPt[0]),int(pxPt[1])), 5, color, 5)
        textLoc = (int(pxPt[0])-30,int(pxPt[1]-30))
        putText(img, str(names[idx]), textLoc, FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 12,
                    LINE_AA)
        putText(img, str(names[idx]), textLoc, FONT_HERSHEY_SIMPLEX, 2,color, 3, LINE_AA)

def project(rvec, tvec, objectPoints, cameraMatrix, distCoeffs):
    projectedPoints, _ = projectPoints(objectPoints, rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix,
                                                distCoeffs=distCoeffs)
    return projectedPoints


#Camera matrix
if paramsOnCalibration:
    # Fix Principle point, aspect ratio, zero tangent distance ON
    orig_fx=1624.4683879211
    orig_fy=1624.4683879211
    orig_cx=711.5000000000
    orig_cy=711.5000000000
else:
    # Fix Principle point, aspect ratio, zero tangent distance OFF
    orig_fx=1547.143
    orig_fy=1534.587
    orig_cx=970.5381732523
    orig_cy=791.7311706726

fx = scale * orig_fx
fy = scale * orig_fy
cx = scale * (orig_cx + 0.5) - 0.5
cy = scale * (orig_cy + 0.5) - 0.5

#Distortion coefficients
if paramsOnDistortion:
    # Fix Principle point, aspect ratio, zero tangent distance ON
    k1=-0.1991660878
    k2=0.2248626435
    p1=0.0000000000
    p2=0.0000000000
    k3=0.4556142974
else:
    # Fix Principle point, aspect ratio, zero tangent distance OFF
    k1=-0.1166279524
    k2=0.0256347102
    p1=0.0233343086
    p2=0.0184426060
    k3=0.0175813388

cameraMatrix = np.eye(3)
cameraMatrix[0,0] = fx
cameraMatrix[1,1] = fy
cameraMatrix[0,2] = cx
cameraMatrix[1,2] = cy

if applyDistortion:
    distCoeffs = np.array([k1, k2, p1, p2, k3])
else:
    distCoeffs = np.zeros((5,))

detector = Detector()

img = imread(aprilImage)
gray = cvtColor(img, COLOR_BGR2GRAY)

detections = detector.detect(gray, estimate_tag_pose=True, camera_params=([fx, fy, cx, cy]), tag_size=tag_size)
centers = None


validPoints = {}
aprilTagPoints = None

# Draw bounding boxes around the detected tags

for detection in detections:
    proj = cameraMatrix @ detection.pose_t
    if aprilTagPoints is None:
        aprilTagPoints = detection.pose_t
    else:
        aprilTagPoints = np.append(aprilTagPoints, detection.pose_t, axis=1)

    pixCenter = (int(detection.center[0]), int(detection.center[1]))

    polylines(img, [detection.corners.astype(int)], True, (0, 255, 0), 2)
    putText(img, str(detection.tag_id), pixCenter,
                FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

    if centers is None:
        centers = np.array(pixCenter)
    else:
        centers = np.vstack((centers, np.array(pixCenter)))

    validPoints[detection.tag_id] = np.array(pixCenter)


truthPointsClass = TruthPoints()
truthPoints = truthPointsClass.getTruthPointsDict()


# objectPoints = objectPoints.astype('float32')
centers = centers.astype('float32')

imagePoints = None

for validPt in validPoints.values():
    if imagePoints is None:
        imagePoints = validPt
    else:
        imagePoints = np.vstack((imagePoints, validPt))

objectPoints = np.zeros((imagePoints.shape[0], 3))
for idx, valID in enumerate(validPoints.keys()):
    objectPoints[idx,:] = truthPoints[str(valID)]

np.set_printoptions(suppress=True)

print('OP: \n', objectPoints.T)
print('Ap Points: \n', aprilTagPoints)
print('IP: \n', centers)
print('CM: \n', cameraMatrix)
print('DP: \n', distCoeffs)

ret, rvec, tvec = solvePnP(objectPoints=objectPoints, imagePoints=centers, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=SOLVEPNP_ITERATIVE)

print('\nRvec: \n', rvec)
print('Rvec as DCM: \n', Rodrigues(rvec)[0])
print('Tvec: \n', tvec)
print('T-norm: \n', la.norm(tvec))

projectedPoints_orig, _ = projectPoints(objectPoints, rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

probeTip_3d = np.array([[4.27289], [-2.50055], [-0.25204]])
probeTip_pix, _ = projectPoints(probeTip_3d, rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

plotOnImg(img, projectedPoints_orig[:,0,:].astype(int), list(validPoints.keys()), (255,255,0))
plotOnImg(img, probeTip_pix[:,0,:].astype(int), ['Probe Tip'], (0,255,0))

putText(img, f'Params On for Calibration: {paramsOnCalibration}', (100,100), FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 15)
putText(img, f'Params On for Calibration: {paramsOnCalibration}', (100,100), FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)

putText(img, f'Params On for Distortion: {paramsOnDistortion}', (100,200), FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 15)
putText(img, f'Params On for Distortion: {paramsOnDistortion}', (100,200), FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)

putText(img, f'Distortion applied: {applyDistortion}', (100,300), FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 15)
putText(img, f'Distortion applied: {applyDistortion}', (100,300), FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)

small_img = resize(img, (848, 848))

imshow("Reproject", small_img)
waitKey(0)
destroyAllWindows()
