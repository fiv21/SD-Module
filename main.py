import time
import os
import sys
import asyncio
from six.moves import input
import cv2
import numpy as np
import scipy.spatial
import json
import logging
import base64 
from azure.iot.device.aio import IoTHubModuleClient
import redis
import queue, threading, multiprocessing
import pyodbc
import datetime
import pandas as pd


ENCODING = 'utf-8'
FORMAT = "[%(asctime)-12s]-[%(levelname)-8s]: %(message)s"

weightsPath = "yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
configPath = "yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

## Get env vars
confid = float(os.getenv('SOCIAL_DISTANCE_confidence', ''))
thresh = float(os.getenv('SOCIAL_DISTANCE_threshold', ''))
DEVICEID = str(os.environ["IOTEDGE_DEVICEID"]) #YPF-PoC
REDIS_HOST=str(os.getenv('REDIS_HOST',''))
REDIS_PORT=str(os.getenv('REDIS_PORT',''))
REDIS_DB=str(os.getenv('REDIS_DB',''))
REDIS_QueueSD=str(os.getenv('REDIS_QueueSD',''))
REDIS_PlotSD=str(os.getenv('REDIS_PlotSD',''))
RES_WIDTH_CAP = int(os.getenv('RES_WIDTH_CAP', ''))
RES_HEIGHT_CAP = int(os.getenv('RES_HEIGHT_CAP', ''))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', ''))
COMPUTING_THREADS = int(os.getenv('COMPUTING_THREADS', ''))
DEBUG_MODE = str(os.getenv('DEBUG_MODE', ''))
SQL_DRIVER = str(os.getenv('SQL_DRIVER',''))
SQL_SERVER = str(os.getenv('SQL_SERVER',''))
SQL_DBNAME = str(os.getenv('SQL_DBNAME',''))
SQL_USERID = str(os.getenv('SQL_USERID',''))
SQL_PASSWD = str(os.getenv('SQL_PASSWD',''))


if DEBUG_MODE == 'True':
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
else:
    logging.basicConfig(format=FORMAT, level=logging.ERROR)


anchorYOLOv3 = 32
anchorSetting = (anchorYOLOv3*13, anchorYOLOv3*13) # 416 x 416 optimal parameter

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

######## MULTI THREAD DEF ###########
q = queue.Queue()
def worker(a): #The guys who will work while the man in charge supervise the operation
    while True:
        f, args = q.get()
        f(*args)
        q.task_done()

threads = []

for i in range(COMPUTING_THREADS): #We create the workers who will do the tasks ASAP
    w = threading.Thread(target=worker, args=(q,))
    w.setDaemon(True)
    #w.daemon = True
    w.start()
    threads.append(w)
#####################################

#####
def connectSQLServer(driver, server, db, user, pwd):
    connSQLServer = pyodbc.connect(
        r'DRIVER={' + driver + '};'
        r'SERVER=' + server + ';'
        r'DATABASE=' + db + ';'
        r'UID=' + user + ';'
        r'PWD=' + pwd + ';'
        r'Trusted_connection=no;',
       autocommit=True
    )
    return connSQLServer


######### for manual procedure, use this functions
######### NOTES: This is not as accurate as the chessboard calibration

def getTransformationMatrix(CamID):
    ## TO DO: Make this function look for a matrix in SQL Server
    testTransformationMatrix = np.load('testingVideoPerspectiveTransformMatrix.npy')
    arr = np.load('testingVideoPerspectiveTransformMatrix_arr.npy')
    return testTransformationMatrix, arr

def getMinSocDist(CamID):
    ## TO DO get social distance from calibration##
    ###########
    socDist = 0.13
    return socDist

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = scipy.spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    dst = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], 
                    image.shape[0]], [0, image.shape[0]]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return M

############### Auxiliar Functions
def dist(c1, c2):
    return (((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2)) ** 0.5

def getDistance(p1,p2, minSocDist):
    distP2P = dist(p1, p2)
    if (distP2P<=minSocDist):
        return 1 #Danger
    if (minSocDist<distP2P and distP2P<=minSocDist*1.5):
        return 2 #Warning
    else:
        return 0 #Safe

def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode(ENCODING)

def base64_decode_image(a, dtype, shape):
    a = bytes(a, encoding=ENCODING)
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a


def getTransformedPoints(centers, CamID, W, H):
    M, arr = getTransformationMatrix(CamID)
    arr = order_points(arr)
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
    return centers


def redisPlot(CamID, ts, p_total, p_safe, p_warning, p_danger, centerX_list, centerY_list, xTr_arr, yTr_arr, status_arr):
    payload = {}
    payload['CamID'] = CamID
    payload['timestamp'] = ts
    payload['safeCount'] = p_safe
    payload['warningCount'] = p_warning
    payload['dangerCount'] = p_danger
    payload['x_CenterPos'] = centerX_list
    payload['y_CenterPos'] = centerY_list
    payload['x_TrPos'] = xTr_arr.tolist()
    payload['y_TrPos'] = yTr_arr.tolist()
    payload['status'] = status_arr
    return json.dumps(payload)


def saveData(CamID, timestamp, peoplePositions, status):
    try:
        sql_conn = connectSQLServer(SQL_DRIVER,SQL_SERVER,SQL_DBNAME,SQL_USERID,SQL_PASSWD)
        cursor = sql_conn.cursor()
        for i in range(len(peoplePositions[0])):
            x_pos = peoplePositions[0][i][0]
            y_pos = peoplePositions[0][i][1]
            color = status[i]
            cursor.execute("INSERT INTO SocialDistance values (?,?,?,?,?)",CamID,timestamp,str(x_pos),str(y_pos),color)
        return "Done!"
    except:
        logging.error("Failed to store in SQL Server!")
        pass
        return False

def socDist(jsonFromRedis):
    CamID = jsonFromRedis["CamID"]
    timestamp = jsonFromRedis["timestamp"]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(jsonFromRedis['encodedImg'].encode(ENCODING)), dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image , 1/255.0, anchorSetting,
                                swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Inference time: {:.6f}".format(end-start))
    boxes = []
    confidences = []
    classIDs = []
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
        dangerList = []
        warningList = []
        center = []
        centerX = []
        centerY = []
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            centerX.append(cen[0])
            centerY.append(cen[1])
            cen = np.array(cen, dtype=np.float32)
            center.append(cen)
            status.append(0)
        transformedPoints = getTransformedPoints(center, CamID, W, H)
        minSocDist = getMinSocDist(CamID)
        if len(transformedPoints[0])>1:
            for i in range(len(transformedPoints[0])-1):
                for j in range(i+1,len(transformedPoints[0])):
                    distStatus = getDistance(transformedPoints[0][i], transformedPoints[0][j], minSocDist)
                    if distStatus == 1:
                        dangerList.append(transformedPoints[0][i]) 
                        dangerList.append(transformedPoints[0][j])
                        status[i] = 1
                        status[j] = 1
                    if distStatus == 2:
                        if status[i] != 1:
                            warningList.append(transformedPoints[0][i])
                            status[i] = 2
                        if status[j] != 1:
                            warningList.append(transformedPoints[0][j])
                            status[j] = 2
        totalPersons = len(transformedPoints[0])
        warningPersons = status.count(2)
        dangerPersons = status.count(1)
        safePersons = status.count(0)
        saveData(CamID, timestamp, transformedPoints, status)
        results = redisPlot(CamID, timestamp, totalPersons, safePersons, warningPersons, 
                  dangerPersons, centerX, centerY, transformedPoints[0,:,0], 
                  transformedPoints[0,:,1], status)
        return results


def redisWatchDog():
    redisQueue = REDIS_QueueSD
    redisConnected = False
    while True:
        if redisConnected==False:
            try:
                db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
                redisConnected = db.ping()
            except:
                redisConnected = False
                pass
            if redisConnected:
                logging.error("Connected to Redis server: {}:{} -> DB: {}".format(REDIS_HOST, REDIS_PORT, REDIS_DB))
                redisConnected=True
            else:
                logging.error("Redis Not available!")
                del db
                time.sleep(1.0)
        else:
            c = 0
            try:
                inMsgCount = db.lrange(redisQueue, 0, BATCH_SIZE)
            except:
                pass
            for msg in inMsgCount:
                data = json.loads(msg.decode(ENCODING))
                results = socDist(data)
                try:
                    db.rpush(REDIS_PlotSD, results)
                    logging.debug(json.dumps(results))
                except:
                     pass
                c += 1
            if c > 0:
                try:
                    db.ltrim(redisQueue, c, -1)
                except:
                    pass
            time.sleep(0.001)


p_capturerMainProc = threading.Thread(target=redisWatchDog)
p_capturerMainProc.daemon = True
p_capturerMainProc.start()

async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        print ( "IoT Hub Client for Python" )
        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()
        # connect the client.
        await module_client.connect()


        # define behavior for receiving an input message on input1
        async def input1_listener(module_client):
            while True:
                input_message = await module_client.receive_message_on_input("input1")  # blocking call
                print("the data in the message received on input1 was ")
                print(input_message.data)
                print("custom properties are")
                print(input_message.custom_properties)
                print("forwarding mesage to output1")
                await module_client.send_message_to_output(input_message, "output1")

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(10)

        # Schedule task for C2D Listener
        listeners = asyncio.gather(input1_listener(module_client))

        print ( "The sample is now waiting for messages. ")

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Cancel listening
        listeners.cancel()

        # Finally, disconnect
        await module_client.disconnect()

    except Exception as e:
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    asyncio.run(main())