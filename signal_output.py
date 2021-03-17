from video_mod import VideoMod
from process_mod import ProcessMod
import os, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class signalWriter(object):
    def __init__(self):

        

        # Path to folder containing videos
        self.dir = './videos'

        self.outputPath = ""
        self.vid_count = 0
        self.process = ProcessMod(GUIMode=False)
        self.current_video_status = True
        self.missed_frame_check = False
        self.write_log_check = False
        self.frame_data = {}
        self.video_data = []
        self.real_data = {}
        self.fake_data = {}
        self.chrom_norm_interval = 9
        self.frame_count = 0
        self.missed_frames = 0
        self.current_missed_streak = 0
        self.greatest_missed_streak = 0
        self.real_count = 0
        self.fake_count = 0
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

    def extractChrominance(self):

        red_R = self.dataset['RC-R'].rolling(self.chrom_norm_interval).mean()
        red_L = self.dataset['LC-R'].rolling(self.chrom_norm_interval).mean()
        red_C = self.dataset['C-R'].rolling(self.chrom_norm_interval).mean()
        red_F = self.dataset['F-R'].rolling(self.chrom_norm_interval).mean()
        red_OR = self.dataset['OR-R'].rolling(self.chrom_norm_interval).mean()
        red_OL = self.dataset['OL-R'].rolling(self.chrom_norm_interval).mean()
        red_CE = self.dataset['CE-R'].rolling(self.chrom_norm_interval).mean()
        green_R = self.dataset['RC-G'].rolling(self.chrom_norm_interval).mean()
        green_L = self.dataset['LC-G'].rolling(self.chrom_norm_interval).mean()
        green_C = self.dataset['C-G'].rolling(self.chrom_norm_interval).mean()
        green_F = self.dataset['F-G'].rolling(self.chrom_norm_interval).mean()
        green_OR = self.dataset['OR-G'].rolling(self.chrom_norm_interval).mean()
        green_OL = self.dataset['OL-G'].rolling(self.chrom_norm_interval).mean()
        green_CE = self.dataset['CE-G'].rolling(self.chrom_norm_interval).mean()
        blue_R = self.dataset['RC-B'].rolling(self.chrom_norm_interval).mean()
        blue_L = self.dataset['LC-B'].rolling(self.chrom_norm_interval).mean()
        blue_C = self.dataset['C-B'].rolling(self.chrom_norm_interval).mean()
        blue_F = self.dataset['F-B'].rolling(self.chrom_norm_interval).mean()
        blue_OR = self.dataset['OR-B'].rolling(self.chrom_norm_interval).mean()
        blue_OL = self.dataset['OL-B'].rolling(self.chrom_norm_interval).mean()
        blue_CE = self.dataset['CE-B'].rolling(self.chrom_norm_interval).mean()


        chrom_R = (1.5 * red_R) - (3 * green_R) + (1.5 * blue_R)
        chrom_L = (1.5 * red_L) - (3 * green_L) + (1.5 * blue_L)
        chrom_C = (1.5 * red_C) - (3 * green_C) + (1.5 * blue_C)
        chrom_F = (1.5 * red_F) - (3 * green_F) + (1.5 * blue_F)
        chrom_OR = (1.5 * red_OR) - (3 * green_OR) + (1.5 * blue_OR)
        chrom_OL = (1.5 * red_OL) - (3 * green_OL) + (1.5 * blue_OL)
        chrom_CE = (1.5 * red_CE) - (3 * green_CE) + (1.5 * blue_CE)



        self.dataset["RC-chrom"] = chrom_R
        self.dataset["LC-Chrom"] = chrom_L
        self.dataset["C-chrom"] = chrom_C
        self.dataset["F-chrom"] = chrom_F
        self.dataset["OR-chrom"] = chrom_OR
        self.dataset["OL-chrom"] = chrom_OL
        self.dataset["CE-chrom"] = chrom_CE





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
                self.frame_data = self.process.frame_data
                self.video_data.append(self.frame_data)
                self.frame_data = {}
            else:
                print("End of Video")
                self.vid_count += 1
                if self.write_log_check:
                    print("Frame(s) missed for file: " + self.full_path)
                    self.write_log()
                self.missed_frame_check = False
                self.current_video_status = False
                video_obj.stop()
                self.dataset = pd.DataFrame.from_records(self.video_data)
                self.extractChrominance()
                if 'fake' in self.full_path.lower():
                    self.dataset['Target'] = 0
                else:
                    self.dataset['Target'] = 1

                processed_file_dir = Path("./processed_new" + self.full_path[1:self.full_path.rindex('/')])
                processed_file_dir.mkdir(parents=True, exist_ok=True)
                self.outputPath = str(processed_file_dir) + self.full_path[self.full_path.rindex('/'):-4] + ".csv"
                self.dataset.to_csv(self.outputPath, index=False)
                self.outputPath = ''
                self.video_data = []
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


