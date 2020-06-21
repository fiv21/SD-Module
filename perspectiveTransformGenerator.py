import time
import math
import cv2
import numpy as np
import os
import os.path
import imutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial


confid = 0.2
thresh = 0.1
anchorYOLOv3 = 32
anchorSetting = (anchorYOLOv3*16, anchorYOLOv3*9)

vname=input("Video path:  ")
if(vname==""):
    vname="videos/cctvMall.mp4"
vid_path = str(vname)

videoResolution = 960 #This is just a numer that works well for my hardware and it's for tests only


angle_factor = 0.4
H_zoom_factor = 0.4

print("VIDEO PATH: {}".format(vid_path))

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)


###### use this for faster processing (caution: slighly lower accuracy) ###########

weightsPath = "yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
configPath = "yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0


### GET THE HOMOGRAPHY POINTS AND THEN MOVE ON ###
refPt = []
cropping = False

def new_order_points(frame, pts):
    arr = [[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]]
    auxarr = []
    for i in range(len(arr)):
        dmin = 10000000000000.0
        for j in range(len(pts)):
            d = scipy.spatial.distance.euclidean(pts[j], arr[i])
            if d < dmin:
                pos = i
                dmin = d
        auxarr.append(pts[pos])
    print(auxarr)
    return auxarr


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 0]), :]
    (tl, bl) = leftMost
    D = scipy.spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(frame, pts):
    decW = frame.shape[1]
    decH = frame.shape[0]
    rect = order_points(pts)
    print(rect)
    (tl, tr, br, bl) = rect
    tl[0] = tl[0]
    tl[1] = tl[1]
    tr[0] = tr[0]
    tr[1] = tr[1]
    br[0] = br[0]
    br[1] = br[1]
    br[0] = br[0]
    bl[1] = bl[1]
    dst = np.array([[0, 0], 
                    [decW, 0], 
                    [decW, decH], 
                    [0, decH]], dtype = "float32")
    M, status = cv2.findHomography(rect, dst)
    return M


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))


def getHomographyPoints(image):
    #we need to improve this applying https://docs.opencv.org/3.4.0/d9/dab/tutorial_homography.html#tutorial_homography_Demo3
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while(len(refPt)<5):
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        cv2.putText(image, "MARK 4 PLANE PERPENDICULAR POINTS | (R) TO RESTART", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r") or key == ord("R"):
            image = clone.copy()
            cv2.destroyAllWindows()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)
        if len(refPt)>1:
            if len(refPt)==4:
                cv2.line(image, refPt[2], refPt[3], (0,255,0), 3)
                cv2.line(image, refPt[3], refPt[0], (0,255,0), 3)
                cv2.putText(image, "4 POINTS MARKED | PRESS (C) TO CONTINUE", (50, image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
                cv2.imshow("image", image)
            else:
                cv2.line(image, refPt[len(refPt)-2], refPt[len(refPt)-1], (0,255,0), 3)
        if (key == ord("c") or key == ord("C")) and len(refPt)==4:
            cv2.destroyAllWindows()
            H = four_point_transform(image, np.array(refPt, dtype="float32"))
            new_order_points(image, np.array(refPt, dtype="float32"))
            return H, (np.array(refPt, dtype="float32"))
    return 0

def generateFrameToMark(videoName):
    command = "ffmpeg -ss 00:00:00.1 -i "+videoName+" -vframes 1 markThisFrame.jpg -y"
    os.system(command)
    return 0

def initialTasks():
    generateFrameToMark(vid_path)#generate a frame to get the 4 points from the plane
                                # This process will change in the future
                                # This should be completely automatic
    frameToMark = cv2.imread('./markThisFrame.jpg') #the easiest way to handle this was using
                                                    #FFMPEG to take the snapshot, save it and then
                                                    #use it with OpenCV
    frameToMark = imutils.resize(frameToMark, width=960, height=540)
    if os.path.isfile(vid_path+'_mtx.npy') and os.path.isfile(vid_path+'_arr.npy'):
        opcion=input("A configuration file exist for this video. Keep config? Y/[n] ")
        if opcion=='y' or opcion == 'Y':
            transformationMatrix = np.load(vid_path+'_mtx.npy')
            print("Transformation Matrix loaded: {}".format(transformationMatrix))
            arr = np.load(vid_path+'_arr.npy')
            print("Transformation points loaded: {}".format(arr))
        else:
            print("Generate a new Transformation Matrix:")
            transformationMatrix, arr = getHomographyPoints(frameToMark) #Here you receive the 3x3 Matrix to make the transformation
                                                            #between the frame and the plane to map the objects 
                                                            #from one field to the other.
                                                            # NOTES: REMEMBER IF YOU INVERT THIS MATRIX AND MAKE THE ANTI-TRANSFORMATION, 
                                                            # YOU SHOULD GET THE SAME RESULTS OR SIMILARS DUE TO THE PRECISION IN DECIMALS
            print("Transformation Matrix generated: {}".format(transformationMatrix))
            np.save(vid_path+'_mtx.npy', transformationMatrix)
            np.save(vid_path+'_arr.npy', arr)
            print("Transformation points loaded: {}".format(arr))
    else:
        print ("Manual configuration required")
        transformationMatrix, arr = getHomographyPoints(frameToMark) #Here you receive the 3x3 Matrix to make the transformation
                                                            #between the frame and the plane to map the objects 
                                                            #from one field to the other.
                                                            # NOTES: REMEMBER IF YOU INVERT THIS MATRIX AND MAKE THE ANTI-TRANSFORMATION, 
                                                            # YOU SHOULD GET THE SAME RESULTS OR SIMILARS DUE TO THE PRECISION IN DECIMALS
        print("Transformation Matrix generated: {}".format(transformationMatrix))
        np.save(vid_path+'_mtx.npy', transformationMatrix)
        np.save(vid_path+'_arr.npy', arr)
        print("Transformation points loaded: {}".format(arr))
    os.system("del markThisFrame.jpg") #delete the generated frame to free the disk
    return transformationMatrix, arr

#############################
# TO DO: save the matrix values, frame used and camera configurations for future analysis
#############################

fig, ax = plt.subplots()

lw = 3
alpha = 0.5
ax.legend()
plt.ion()
plt.show()

#### FROM HERE WE CALCULATE THE SOCIAL DISTANCE

def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    a_w = (p1[0]+p2[0])/2
    a_h = (p1[1]+p2[1])/2
    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0




transformationMatrix, extremePoints=initialTasks()

vs = cv2.VideoCapture(vid_path)
cv2.waitKey(100) 

_, img = vs.read()
img = imutils.resize(img, width=960, height=540)
newpts = new_order_points(img, extremePoints)

def getTransformedPoints(centers):
    M = transformationMatrix
    arr = newpts
    extremes = cv2.perspectiveTransform(np.array([arr]), M)
    xy0 = [np.amin(extremes[:,:,0]), np.amin(extremes[:,:,1]), np.amax(extremes[:,:,0]), np.amax(extremes[:,:,1])]
    xmin = xy0[0]
    xmax = xy0[2]
    ymin = xy0[1]
    ymax = xy0[3]
    deltaX = xmax - xmin
    deltaY = ymax - ymin
    centers = cv2.perspectiveTransform(np.array([centers]), M)
    for i in range(len(centers[0])):
        centers[0][i][0] = (centers[0][i][0]-xmin)/deltaX
        centers[0][i][1] = (centers[0][i][1]-ymin)/deltaY
    print("Normalized data: {}".format(centers))
    return centers

# print("Graph Limits: {}".format(cv2.perspectiveTransform(np.array([np.array([[0,0],[img.shape[1],0],[img.shape[1],img.shape[0]],[0,img.shape[0]]], dtype='float32')]), transformationMatrix)))
# extremes = getTransformedPoints(extremePoints)
# xy0 = [np.amin(extremes[:,:,0]), np.amin(extremes[:,:,1]), np.amax(extremes[:,:,0]), np.amax(extremes[:,:,1])]
xy0 = [0.0,0.0,1.0,1.0]




writer = None
(W, H) = (None, None)



while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    frame = imutils.resize(frame, width=960, height=540)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:] = col

    realTimeScatterPlot=np.zeros((frame.shape[0],FW-frame.shape[1],3), np.uint8)


    blob = cv2.dnn.blobFromImage(frame , 1/255.0, anchorSetting,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []
    pointsToScatter = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs =  cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            normCen = np.asarray(cen, dtype="float32")
            pointsToScatter.append(normCen)
            cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])
                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2
        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0



        for i in idf:
            cv2.line(FR,(0,H+1),(FW,H+1),(0,0,0),2)
            cv2.putText(FR, "Social Distancing", (210, H+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(FR, (20, H+80), (510, H+180), (100, 100, 100), 2)
            cv2.putText(FR, "Las lineas conectadas representan la cercania de las personas. ", (30, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "--Amarillo: Cerca", (50, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 170), 2)
            cv2.putText(FR, "--Rojo: Muy Cerca (Peligroso)", (50, H+40+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.rectangle(FR, (535, H+80), (1060, H+140+40), (100, 100, 100), 2)
            cv2.putText(FR, "Lo que marca a la person indica el nivel de riesgo.", (545, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "--Rojo: Alto Riesgo", (565, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
            cv2.putText(FR, "--Naranja: Mediano Riesgo", (565, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)

            cv2.putText(FR, "--Verde: Bajo", (565, H+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

            
            tot_str = "TOTAL: " + str(total_p)
            high_str = "ALTO RIESGO: " + str(high_risk_p)
            low_str = "MEDIANO RIESGO: " + str(low_risk_p)
            safe_str = "BAJO RIESGO: " + str(safe_p)

            cv2.putText(FR, tot_str, (10, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(FR, safe_str, (200, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 0), 2)
            cv2.putText(FR, low_str, (380, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            cv2.putText(FR, high_str, (630, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1

        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
        
        FR[0:H, 0:W] = frame
        #FR[H:FR.shape[0],W:FR.shape[1]] = realTimeScatterPlot
        frame = FR
        cv2.imshow('Social distancing analyser', frame)
        cv2.waitKey(1)
        plotScatter = getTransformedPoints(pointsToScatter)
        fig.canvas.flush_events()
        ax.set_xlim(xy0[0], xy0[2])
        ax.set_ylim(xy0[1], xy0[3])
        ax.set_title('Mapped Persons')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.scatter(plotScatter[:,:,0], plotScatter[:,:,1], color='green')
        fig.canvas.draw()
        plt.draw()
#################
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("outputSD.avi", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
print("Processing finished: open"+" op_"+vname)
if writer is not None:
    writer.release()
vs.release()
