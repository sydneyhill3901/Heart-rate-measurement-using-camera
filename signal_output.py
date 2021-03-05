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


    def write_log(self):
        print("log")
        self.file_log['File'] = self.full_path
        self.file_log['Frames missed'] = self.missed_frames
        self.file_log['Largest missing interval'] = self.greatest_missed_streak
        self.file_log['Percent detected'] = (self.frame_count - self.missed_frames)/self.frame_count
        self.writer.writerow(self.file_log)

    def write_derived_features(self):

        difg_LCRC = (self.dataset['RC-G'] - self.dataset['LC-G']).abs()
        difc_LCRC = (self.dataset['RC-C'] - self.dataset['LC-C']).abs()
        difg_LCRC.iloc[:200].to_csv(self.dir + '/difgLR.csv')
        difc_LCRC.iloc[:200].to_csv(self.dir + '/difcLR.csv')
        difg_o = ((self.dataset['C-G'] - self.dataset['F-G']).abs() - (self.dataset['OR-G'] - self.dataset['OL-G']).abs()).abs()
        difc_o = ((self.dataset['C-C'] - self.dataset['F-C']).abs() - (self.dataset['OR-C'] - self.dataset['OL-C']).abs()).abs()
        difg_o.iloc[:200].to_csv(self.dir + '/difg_o.csv')
        difc_o.iloc[:200].to_csv(self.dir + '/difc_o.csv')
        derived_data = difg_LCRC.iloc[:200].append(difc_LCRC.iloc[:200], ignore_index=True).append(difg_o.iloc[:200], ignore_index=True).append(
            difc_o.iloc[:200], ignore_index=True)

        if 'fake' in self.full_path.lower():
            self.fake_data[self.fake_count] = derived_data
            self.fake_count += 1
        else:
            self.real_data[self.real_count] = derived_data
            self.real_count += 1

        self.vid_count += 1


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
                self.write_derived_features()
                if 'fake' in self.full_path.lower():
                    self.dataset['Target'] = 0
                else:
                    self.dataset['Target'] = 1

                processed_file_dir = Path("./processed" + self.full_path[1:self.full_path.rindex('/')])
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


        self.real_df = pd.DataFrame(self.real_data)
        self.fake_df = pd.DataFrame(self.fake_data)
        r = self.real_df.transpose()
        f = self.fake_df.transpose()
        r['Target'] = 1
        f['Target'] = 0
        rf = r.append(f)
        rf.to_csv(self.dir + '/all_data.csv', index=False)



p = signalWriter()


