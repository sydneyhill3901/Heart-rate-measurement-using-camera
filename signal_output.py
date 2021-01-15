from video_mod import VideoMod
from process_mod import ProcessMod
import os, csv
import numpy as np
import pandas as pd
from pathlib import Path

class signalWriter(object):
    def __init__(self):
        self.dir = './videos'
        self.outputPath = ""
        self.process = ProcessMod(GUIMode=False)
        self.current_video_status = True
        self.missed_frame_check = False
        self.write_log_check = False
        self.frameData = {}
        self.videoData = []
        self.frame_count = 0
        self.missed_frames = 0
        self.current_missed_streak = 0
        self.greatest_missed_streak = 0
        self.full_path = ''
        self.log_path = self.dir + '/process_log.csv'
        self.output = open(self.log_path, "w+", encoding="utf8")
        self.writer = csv.DictWriter(self.output, fieldnames=['File', 'Frames missed', 'Largest missing interval', 'Percent detected'], delimiter=',', lineterminator="\r", quoting=csv.QUOTE_NONE)
        self.file_log = {
                            'File': '',
                            'Frames missed': '',
                            'Largest missing interval': '',
                            'Percent detected': ''
                        }

        self.main()


    def write_log(self):
            print("log")
            self.file_log['File'] = self.full_path
            self.file_log['Frames missed'] = self.missed_frames
            self.file_log['Largest missing interval'] = self.greatest_missed_streak

            self.file_log['Percent detected'] = (self.frame_count - self.missed_frames)/self.frame_count
            self.writer.writerow(self.file_log)



    def main_loop(self, video_obj):

        frame = video_obj.get_frame()
        self.process.frame_in = frame
        try:
            if frame is not None:
                try:
                    self.process.run()
                    self.missed_frame_check = False
                except:
                    self.missed_frame_check = True
                    self.write_log_check = True
                    self.missed_frames += 1
                    self.current_missed_streak += 1
                if self.missed_frame_check:
                    if self.current_missed_streak > self.greatest_missed_streak:
                        self.greatest_missed_streak = self.current_missed_streak
                else:
                    self.current_missed_streak = 0

                self.frame_count += 1
                self.frameData = self.process.frameData
                self.videoData.append(self.frameData)
                self.frameData = {}
            else:
                print("End of Video")
                if self.write_log_check:
                    print("Frame(s) missed for file: " + self.full_path)
                    self.write_log()
                self.missed_frame_check = False
                self.current_video_status = False
                video_obj.stop()
                dataset = pd.DataFrame.from_records(self.videoData)
                if 'fake' in self.full_path.lower():
                    dataset['Target'] = 0
                else:
                    dataset['Target'] = 1

                processed_file_dir = Path("./processed" + self.full_path[1:self.full_path.rindex('/')])
                processed_file_dir.mkdir(parents=True, exist_ok=True)
                self.outputPath = str(processed_file_dir) + self.full_path[self.full_path.rindex('/'):-4] + ".csv"
                dataset.to_csv(self.outputPath, index=False)
                self.outputPath = ''
                return

        except Exception as excep:
            print("Exception occurred in main_loop: " + str(excep))


    def main(self):

        video_obj = VideoMod()
        self.writer.writeheader()
        for paths, subdir, files in os.walk(self.dir):
            for file in files:
                self.frame_count = 0
                self.missed_frames = 0
                self.series_missed = 0
                self.write_log_check = False
                if file.endswith('.mp4') or file.endswith('.avi'):
                    try:
                        self.full_path = os.path.join(paths, file)
                        self.full_path = self.full_path.replace("\\", "/")
                        video_obj.filename = self.full_path
                        video_obj.start()
                        self.current_video_status = True  # if video_obj.start() executed successfully, then status is True
                        print('Processing file: ' + self.full_path)
                        while self.current_video_status == True:
                            self.main_loop(video_obj)


                    except Exception as excep:
                        print("Exception occurred on file " + file + ": " + str(excep))


p = signalWriter()


