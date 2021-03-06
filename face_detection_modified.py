import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from face_recognition import api
import time

class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)
        self.facecount = 0
        self.framecount = 0
        self.starttime = 0

    def face_detect(self, frame):

        frame = imutils.resize(frame, width=1000)
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        status = False

        if frame is None:
            return
        frame_shape_x = frame.shape[1]
        frame_shape_y = frame.shape[0]
        rgb_frame = frame[:, :, ::-1]
        # detect faces in the rgb image
        rects = api.face_detector(rgb_frame)
        xscale = 640 / frame_shape_x
        yscale = 480 / frame_shape_y



        if len(rects)>0:
            status = True

            #(x, y, w, h) = face_utils.rect_to_bb(rects[0])

            #if y<0:
                #print("a")
                #return frame, face_frame, ROI1, ROI2, status, mask

            #face_frame = frame[y:y+h,x:x+w]

            #if(face_frame.shape[:2][1] != 0):
                #face_frame = imutils.resize(face_frame,width=256)
                
            face_frame = self.fa.align(frame,rgb_frame,rects[0])
            rgb_frame_f = face_frame[:, :, ::-1]
            #rectsf = api.face_detector(rgb_frame_f)
            r = dlib.rectangle(0, 0, 256, 256)
            if len(rects) > 0:

                shape = self.predictor(rgb_frame_f, r)
                shape = face_utils.shape_to_np(shape)

                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks

                #cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 0, 255), 1)
                scalefactor = shape[27][1] - shape[30][1]

                cv2.rectangle(face_frame,(shape[54][0], shape[35][1]), # right cheek
                        ((shape[12][0] + int(scalefactor/5)), shape[65][1]), (0,255,0), 0)
                cv2.rectangle(face_frame, ((shape[4][0] - int(scalefactor/5)), shape[35][1]), # left cheek
                        (shape[48][0], shape[65][1]), (0,255,0), 0)
                cv2.rectangle(face_frame, (shape[59][0], (shape[57][1] - int(scalefactor/4))), # chin/neck
                        (shape[55][0],(shape[8][1] - int(scalefactor/3))), (0,255,0), 0)
                cv2.rectangle(face_frame, (shape[18][0], (shape[18][1] + scalefactor)), # forehead
                              (shape[25][0],(shape[24][1] + int(scalefactor/3))), (0,255,0), 0)
                #cv2.rectangle(face_frame, ((shape[12][0] + int(scalefactor/4)), shape[16][1]),  # outer right
                #              ((shape[12][0] - int(scalefactor/6)), shape[13][1]), (0, 255, 0), 0)
                #cv2.rectangle(face_frame, ((shape[4][0] - int(scalefactor/4)), shape[0][1]),  # outer left
                #              ((shape[4][0] + int(scalefactor/6)), shape[3][1]), (0, 255, 0), 0)
                cv2.rectangle(face_frame, (shape[54][0], shape[28][1]),  # center
                              (shape[48][0], shape[35][1]), (0, 255, 0), 0)
                
                ROI1 = face_frame[shape[35][1]:shape[65][1], #right cheek
                       shape[54][0]:(shape[12][0] + int(scalefactor/5))]
                        
                ROI2 =  face_frame[shape[35][1]:shape[65][1], #left cheek
                        (shape[4][0] - int(scalefactor/5)):shape[48][0]]

                ROI3 = face_frame[(shape[57][1] - int(scalefactor/4)):(shape[8][1] - int(scalefactor/3)), # chin/neck
                        shape[59][0]:shape[55][0]]

                ROI4 = face_frame[(shape[18][1] + scalefactor):(shape[24][1] + int(scalefactor/3)), # forehead
                        shape[18][0]:shape[25][0]]

                ROI5 = face_frame[shape[16][1]:shape[13][1],
                       (shape[12][0] + int(scalefactor/4)):(shape[12][0] - int(scalefactor/6))]

                ROI6 = face_frame[shape[0][1]:shape[3][1],
                       (shape[4][0] + int(scalefactor/6)):(shape[4][0] - int(scalefactor/4))]

                ROI7 = face_frame[shape[28][1]:shape[35][1], # center
                        shape[48][0]:shape[54][0]]

                rshape = np.zeros_like(shape) 
                rshape = self.face_remap(shape)


                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))
            
                cv2.fillConvexPoly(mask, rshape[0:27], 1)

        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False

        frame = cv2.resize(frame, (640, 480))
        return frame, face_frame, ROI1, ROI2, ROI3, ROI4, ROI5, ROI6, ROI7, status, mask

    # some points in the facial landmarks need to be re-ordered
    def face_remap(self, shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]
        
        remapped_image = cv2.convexHull(shape)
        return remapped_image       
        
        
        
        
        
