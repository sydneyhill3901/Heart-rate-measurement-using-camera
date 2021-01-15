import cv2
import numpy as np
import time
from face_detection_modified import FaceDetection
from scipy import signal


class ProcessMod(object):
    def __init__(self, GUIMode=False):
        self.calculateHR = GUIMode
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100 green component
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.process_this_frame = True
        self.framecount = 0
        self.facecount = 0


    def extractColor(self, frame):

        g = np.mean(frame[:, :, 1])
        return g

    # https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
    def skinMask1(self, frame, tolerance=4):

        tolerance = 9
        converted_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        converted1 = cv2.cvtColor(self.ROI1, cv2.COLOR_BGR2HSV)
        converted2 = cv2.cvtColor(self.ROI2, cv2.COLOR_BGR2HSV)
        avg1 = np.mean(np.mean(converted1, axis=1), axis=0)
        avg2 = np.mean(np.mean(converted2, axis=1), axis=0)
        avg = (avg1 + avg2) / 2
        dev1 = np.std(np.std(converted1, axis=1), axis=0)
        dev2 = np.std(np.std(converted2, axis=1), axis=0)
        dev = (dev1 + dev2) / 2
        lower = avg - (dev * tolerance) - [5, 5, 5]
        upper = avg + (dev * tolerance) + [5, 5, 5]
        skinMask = cv2.inRange(converted_frame, lower, upper)
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        return skinMask

    # https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py
    def skinMask2(self, frame):

        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        return global_mask

    def skinMask3(self, frame):

        mask = cv2.bitwise_or(self.skinMask1(frame), self.skinMask2(frame), self.mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
        return mask

    def extractChrominance(self, frame):

        fBlue = np.mean(frame[1, :, :])
        fRed = np.mean(frame[:, : ,2])
        fGreen = np.mean(frame[:, :, 0])
        rChrom = fRed / (fRed + fBlue + fGreen)
        gChrom = fGreen / (fRed + fBlue + fGreen)
        return rChrom, gChrom

    def run(self):

        applyMask = 3

        if self.process_this_frame:
            try:
                self.frame, self.face_frame, self.ROI1, self.ROI2, self.ROI3, self.ROI4, self.ROI5, self.ROI6, self.status, self.mask = self.fd.face_detect(
                    self.frame_in)
            except TypeError:
                print("end video")
        self.process_this_frame = not self.process_this_frame

        self.framecount += 1
        if self.status:
            self.facecount += 1

        face_frame = self.face_frame
        status = self.status
        mask = self.mask

        if applyMask == 1:
            try:
                self.frame = cv2.bitwise_and(self.frame, self.frame, mask=self.skinMask1(self.frame))
            except cv2.error:
                print("frame error")
        elif applyMask == 2:
            try:
                self.frame = cv2.bitwise_and(self.frame, self.frame, mask=self.skinMask2(self.frame))
            except cv2.error:
                print("frame error")
        elif applyMask == 3:
            try:
                self.frame = cv2.bitwise_and(self.frame, self.frame, mask=self.skinMask3(self.frame))
            except cv2.error:
                print("frame error")


        self.frame_out = self.frame
        self.frame_ROI = self.face_frame


        g1 = self.extractColor(self.ROI1)
        g2 = self.extractColor(self.ROI2)


        if self.calculateHR:

            g = (g1 + g2) / 2

            L = len(self.data_buffer)

            if (abs(g - np.mean(
                    self.data_buffer)) > 10 and L > 99):  # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
                g = self.data_buffer[-1]

            self.times.append(time.time() - self.t0)
            self.data_buffer.append(g)

            # only process in a fixed-size buffer
            if L > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                self.bpms = self.bpms[-self.buffer_size // 2:]
                L = self.buffer_size

            processed = np.array(self.data_buffer)

            # start calculating after the first 10 frames
            if L == self.buffer_size:
                self.fps = float(L) / (self.times[-1] - self.times[0])
                print("fps: " + str(self.fps))
                # calculate HR using a true fps of processor of the computer, not the fps the camera provide
                even_times = np.linspace(self.times[0], self.times[-1], L)

                processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
                interpolated = np.interp(even_times, self.times, processed)  # interpolation by 1
                interpolated = np.hamming(
                    L) * interpolated  # make the signal become more periodic (advoid spectral leakage)
                # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
                norm = interpolated / np.linalg.norm(interpolated)
                raw = np.fft.rfft(norm * 30)  # do real fft with the normalization multiplied by 10

                self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
                freqs = 60. * self.freqs

                # idx_remove = np.where((freqs < 50) & (freqs > 180))
                # raw[idx_remove] = 0

                self.fft = np.abs(raw) ** 2  # get amplitude spectrum

                idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
                pruned = self.fft[idx]
                pfreq = freqs[idx]

                self.freqs = pfreq
                self.fft = pruned

                idx2 = np.argmax(pruned)  # max in the range can be HR

                self.bpm = self.freqs[idx2]
                self.bpms.append(self.bpm)

                processed = self.butter_bandpass_filter(processed, 0.8, 3, self.fps, order=3)
                # ifft = np.fft.irfft(raw)
            self.samples = processed  # multiply the signal with 5 for easier to see in the plot
            # TODO: find peaks to draw HR-like signal.

            if (mask.shape[0] != 10):
                out = np.zeros_like(face_frame)
                mask = mask.astype(np.bool)
                out[mask] = face_frame[mask]
                if (processed[-1] > np.mean(processed)):
                    out[mask, 2] = 180 + processed[-1] * 10
                face_frame[mask] = out[mask]
        else:

            g3 = self.extractColor(self.ROI3)
            g4 = self.extractColor(self.ROI4)
            g5 = self.extractColor(self.ROI5)
            g6 = self.extractColor(self.ROI6)

            cr1, cg1 = self.extractChrominance(self.ROI1)
            cr2, cg2 = self.extractChrominance(self.ROI2)
            cr3, cg3 = self.extractChrominance(self.ROI3)
            cr4, cg4 = self.extractChrominance(self.ROI4)
            cr5, cg5 = self.extractChrominance(self.ROI5)
            cr6, cg6 = self.extractChrominance(self.ROI6)

            self.frameData = {
                'RC-G' : g1,
                'RC-Cr' : cr1,
                'RC-Cg' : cg1,
                'LC-G' : g2,
                'LC-Cr' : cr2,
                'LC-Cg' : cg2,
                'C-G' : g3,
                'C-Cr' : cr3,
                'C-Cg' : cg3,
                'F-G' : g4,
                'F-Cr' : cr4,
                'F-Cg' : cg4,
                'OR-G' : g5,
                'OR-Cr' : cr5,
                'OR-Cg' : cg5,
                'OL-G' : g6,
                'OL-Cr' : cr6,
                'OL-Cg' : cg6}


    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y











