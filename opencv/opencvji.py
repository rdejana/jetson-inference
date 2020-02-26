import numpy as np
import cv2
import jetson.inference
import jetson.utils

#
# Requires that https://github.com/dusty-nv/jetson-inference be installed.
#
#

# setup the network we are using
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# start our camera.  USB came may be 0 or 1
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    # useful link for going the other way https://github.com/dusty-nv/jetson-inference/issues/356

    ret, frame = cap.read()
    w = frame.shape[1]
    h = frame.shape[0]
    #to RGBA
    # temp = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    # need to go to float 32
    input_image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA).astype(np.float32)
    #Next, we have to move the image to CUDA:
    input_image = jetson.utils.cudaFromNumpy(input_image)
  
    detections = net.Detect(input_image, w, h)
    print("detected {:d} objects in image".format(len(detections)))
    for detection in detections:
        print(detection)
    # Display the resulting frame
    numpyImg  = jetson.utils.cudaToNumpy(input_image,w, h,4)
    #print("n -> ", numpyImg.dtype)
    # now back to unit8
    tt = numpyImg.astype(np.uint8)
    cv2.imshow('frame',tt)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
