import os, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import signal

class ProcessSignalData(object):
    def __init__(self):

        # path to video data from signal_output.py
        self.dir = './processed_new/videos'
        self.full_path = ''
        self.dataframe = pd.DataFrame()
        self.real_data = pd.DataFrame()
        self.fake_data = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.real_data_mean = {}
        self.fake_data_mean = {}
        self.real_data_var = {}
        self.fake_data_var = {}
        self.real_data_std = {}
        self.fake_data_std = {}
        self.real_data_psd = {}
        self.fake_data_psd = {}
        self.real_data_csd = {}
        self.fake_data_csd = {}
        self.real_data_f1 = {}
        self.fake_data_f1 = {}
        self.real_data_test = {}
        self.fake_data_test = {}
        self.real_data_RCCE = {}
        self.real_data_LCCE = {}
        self.real_data_LCRC = {}
        self.fake_data_RCCE = {}
        self.fake_data_LCCE = {}
        self.fake_data_LCRC = {}
        self.real_count = 0
        self.fake_count = 0
        self.vid_count = 0
        self.data_path_lcce = './lcce250.csv'
        self.data_path_lcrc = './lcrc250.csv'
        self.data_path_rcce = './rcce250.csv'
        self.data_path_m = './mean_data16.csv'
        self.data_path_v = './new_chrom/var_data16.csv'
        self.data_path_s = './new_chrom/std_data16.csv'
        self.data_path_p = './new_chrom/psd_data16.csv'
        self.data_path_c = './new_chrom/csd_data_128.csv'
        self.data_path_c = './f1_data_128.csv'
        self.log_path = './process_log.csv'
        self.test_data_lcce_path = './new_chrom/test_lcce.csv'
        self.test_data_lcrc_path = './new_chrom/test_lcrc.csv'
        self.test_data_rcce_path = './new_chrom/test_rcce.csv'
        self.train_data_lcce_path = './new_chrom/train_lcce.csv'
        self.train_data_lcrc_path = './new_chrom/train_lcrc.csv'
        self.train_data_rcce_path = './new_chrom/train_rcce.csv'
        self.test_data_v_path = './new_chrom/train_data_v32c.csv'
        self.train_data_v_path = './new_chrom/test_data_v32c.csv'
        self.test_data_m_path = './new_chrom/train_data_m32c.csv'
        self.train_data_m_path = './new_chrom/test_data_m32c.csv'
        self.test_data_s_path = './new_chrom/train_data_s32c.csv'
        self.train_data_s_path = './new_chrom/test_data_s32c.csv'
        self.test_data_p_path = './new_chrom/train_data_p128c.csv'
        self.train_data_p_path = './new_chrom/test_data_p128c.csv'
        self.test_data_c_path = './train_data_c128c.csv'
        self.train_data_c_path = './test_data_c128c.csv'
        self.test_data_f1_path = './train_data_f1-128c.csv'
        self.train_data_f1_path = './test_data_f1-128c.csv'
        self.test_data_test_path = './train_data_test.csv'
        self.train_data_test_path = './test_data_test.csv'

        self.main()

    def new_chrom(self, red, green, blue):

        # calculation of new X and Y
        Xcomp = 3 * red - 2 * green
        Ycomp = (1.5 * red) + green - (1.5 * blue)

        # standard deviations
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)

        alpha = sX / sY

        # -- rPPG signal
        bvp = Xcomp - alpha * Ycomp

        return bvp

    def main(self):

        # length of video in frames to process
        sample_length = 250

        # interval for mean, var, std
        group_size = 32

        #window for psd
        psd_size = 128

        for paths, subdir, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.csv'):
                    self.full_path = os.path.join(paths, file)
                if 'rejected' in self.full_path.lower() or '.txt' in self.full_path.lower() or 'imposter' in self.full_path.lower():
                    pass
                else:
                    print(self.full_path)
                    self.dataset = pd.read_csv(self.full_path)

                    right_R = self.dataset['RC-R'].iloc[:sample_length]
                    left_R = self.dataset['LC-R'].iloc[:sample_length]
                    chin_R = self.dataset['C-R'].iloc[:sample_length]
                    forehead_R = self.dataset['F-R'].iloc[:sample_length]
                    outerR_R = self.dataset['OR-R'].iloc[:sample_length]
                    outerL_R = self.dataset['OL-R'].iloc[:sample_length]
                    center_R = self.dataset['CE-R'].iloc[:sample_length]

                    right_G = self.dataset['RC-G'].iloc[:sample_length]
                    left_G = self.dataset['LC-G'].iloc[:sample_length]
                    chin_G = self.dataset['C-G'].iloc[:sample_length]
                    forehead_G = self.dataset['F-G'].iloc[:sample_length]
                    outerR_G = self.dataset['OR-G'].iloc[:sample_length]
                    outerL_G = self.dataset['OL-G'].iloc[:sample_length]
                    center_G = self.dataset['CE-G'].iloc[:sample_length]

                    right_B = self.dataset['RC-B'].iloc[:sample_length]
                    left_B = self.dataset['LC-B'].iloc[:sample_length]
                    chin_B = self.dataset['C-B'].iloc[:sample_length]
                    forehead_B = self.dataset['F-B'].iloc[:sample_length]
                    outerR_B = self.dataset['OR-B'].iloc[:sample_length]
                    outerL_B = self.dataset['OL-B'].iloc[:sample_length]
                    center_B = self.dataset['CE-B'].iloc[:sample_length]

                    right_C = self.dataset['RC-chrom'].iloc[:sample_length]
                    left_C = self.dataset['LC-Chrom'].iloc[:sample_length]
                    chin_C = self.dataset['C-chrom'].iloc[:sample_length]
                    forehead_C = self.dataset['F-chrom'].iloc[:sample_length]
                    outerR_C = self.dataset['OR-chrom'].iloc[:sample_length]
                    outerL_C = self.dataset['OL-chrom'].iloc[:sample_length]
                    center_C = self.dataset['CE-chrom'].iloc[:sample_length]

                    chrom_R = right_C
                    chrom_L = left_C
                    chrom_CE = center_C
                    chrom_OL = outerL_C
                    chrom_OR = outerR_C

                    #chrom_R = self.new_chrom(right_R, right_G, right_B)
                    #chrom_L = self.new_chrom(left_R, left_G, left_B)
                    chrom_C = self.new_chrom(chin_R, chin_G, chin_B)
                    chrom_F = self.new_chrom(forehead_R, forehead_G, forehead_B)
                    #chrom_OR = self.new_chrom(outerR_R, outerR_G, outerR_B)
                    #chrom_OL = self.new_chrom(outerL_R, outerL_G, outerL_B)
                    #chrom_CE = self.new_chrom(center_R, center_G, center_B)

                    difg_LCRC = (self.dataset['RC-G'].iloc[:sample_length] - self.dataset['LC-G'].iloc[:sample_length]).abs()
                    difc_LCRC = (self.dataset['RC-chrom'].iloc[:sample_length] - self.dataset['LC-Chrom'].iloc[:sample_length]).abs()
                    difg_o1 = (self.dataset['C-G'].iloc[:sample_length] - self.dataset['F-G'].iloc[:sample_length]).abs()
                    difc_o1 = (self.dataset['C-chrom'].iloc[:sample_length] - self.dataset['F-chrom'].iloc[:sample_length]).abs()
                    difg_o2 = (self.dataset['OR-G'].iloc[:sample_length] - self.dataset['OL-G'].iloc[:sample_length]).abs()
                    difc_o2 = (self.dataset['OR-chrom'].iloc[:sample_length] - self.dataset['OL-chrom'].iloc[:sample_length]).abs()

                    difc_LCCe = (self.dataset['LC-Chrom'].iloc[:sample_length] - self.dataset['CE-chrom'].iloc[
                                                                                 :sample_length]).abs()
                    difc_RCCe = (self.dataset['RC-chrom'].iloc[:sample_length] - self.dataset['CE-chrom'].iloc[
                                                                                 :sample_length]).abs()

                    difc_LCRC = (chrom_R.iloc[:sample_length] - chrom_L.iloc[:sample_length]).abs()
                    difc_LCCe = (chrom_L.iloc[:sample_length] - chrom_CE.iloc[:sample_length]).abs()
                    difc_RCCe = (chrom_R.iloc[:sample_length] - chrom_CE.iloc[:sample_length]).abs()
                    difc_LCOL = (chrom_L.iloc[:sample_length] - chrom_OL.iloc[:sample_length]).abs()
                    difc_RCOR = (chrom_R.iloc[:sample_length] - chrom_OR.iloc[:sample_length]).abs()

                    difg_LCOL = (self.dataset['LC-G'].iloc[:sample_length] - self.dataset['OL-G'].iloc[:sample_length]).abs()
                    difg_RCOR = (self.dataset['RC-G'].iloc[:sample_length] - self.dataset['OR-G'].iloc[:sample_length]).abs()

                    # green channel features

                    # right cheek - left cheek
                    difg_LCRC_lst = [difg_LCRC.iloc[i:i + group_size] for i in
                                     range(0, len(difg_LCRC) - group_size + 1, group_size)]
                    # forehead - chin
                    difg_o1_lst = [difg_o1.iloc[i:i + group_size] for i in
                                   range(0, len(difg_o1) - group_size + 1, group_size)]
                    # outer right - outer left
                    difg_o2_lst = [difg_o2.iloc[i:i + group_size] for i in
                                   range(0, len(difg_o2) - group_size + 1, group_size)]

                    # chrominance features

                    # right cheek - left cheek
                    difc_LCRC_lst = [difc_LCRC.iloc[i:i + group_size] for i in
                                     range(0, len(difc_LCRC) - group_size + 1, group_size)]
                    # forehead - chin
                    difc_o1_lst = [difc_o1.iloc[i:i + group_size] for i in
                                   range(0, len(difc_o1) - group_size + 1, group_size)]
                    # outer right - outer left
                    difc_o2_lst = [difc_o2.iloc[i:i + group_size] for i in
                                   range(0, len(difc_o2) - group_size + 1, group_size)]

                    # mean
                    difg_LCRC_mean = np.array([difg_LCRC_lst[i].mean() for i in range(len(difg_LCRC_lst))])
                    difc_LCRC_mean = np.array([difc_LCRC_lst[i].mean() for i in range(len(difc_LCRC_lst))])
                    print("MEAN")
                    print(difc_LCRC_mean)
                    difg_o1_mean = np.array([difg_o1_lst[i].mean() for i in range(len(difg_o1_lst))])
                    difc_o1_mean = np.array([difc_o1_lst[i].mean() for i in range(len(difc_o1_lst))])
                    difg_o2_mean = np.array([difg_o2_lst[i].mean() for i in range(len(difg_o2_lst))])
                    difc_o2_mean = np.array([difc_o2_lst[i].mean() for i in range(len(difc_o2_lst))])

                    # variance
                    difg_LCRC_var = np.array([difg_LCRC_lst[i].var() for i in range(len(difg_LCRC_lst))])
                    difc_LCRC_var = np.array([difc_LCRC_lst[i].var() for i in range(len(difc_LCRC_lst))])
                    print("VAR")
                    print(difc_LCRC_var)
                    difg_o1_var = np.array([difg_o1_lst[i].var() for i in range(len(difg_o1_lst))])
                    difc_o1_var = np.array([difc_o1_lst[i].var() for i in range(len(difc_o1_lst))])
                    difg_o2_var = np.array([difg_o2_lst[i].var() for i in range(len(difg_o2_lst))])
                    difc_o2_var = np.array([difc_o2_lst[i].var() for i in range(len(difc_o2_lst))])

                    # standard deviation
                    difg_LCRC_std = np.array([difg_LCRC_lst[i].std() for i in range(len(difg_LCRC_lst))])
                    difc_LCRC_std = np.array([difc_LCRC_lst[i].std() for i in range(len(difc_LCRC_lst))])
                    print("STD")
                    print(difc_LCRC_std)
                    difg_o1_std = np.array([difg_o1_lst[i].std() for i in range(len(difg_o1_lst))])
                    difc_o1_std = np.array([difc_o1_lst[i].std() for i in range(len(difc_o1_lst))])
                    difg_o2_std = np.array([difg_o2_lst[i].std() for i in range(len(difg_o2_lst))])
                    difc_o2_std = np.array([difc_o2_lst[i].std() for i in range(len(difc_o2_lst))])

                    # power spectral density
                    f, difg_LCRC_psd = signal.welch(difg_LCRC, nperseg=psd_size)
                    f, difc_LCCe_psd = signal.welch(difc_LCCe, nperseg=psd_size)
                    f, difc_RCCe_psd = signal.welch(difc_RCCe, nperseg=psd_size)
                    f, difc_LCRC_psd = signal.welch(difc_LCRC, nperseg=psd_size)
                    print("PSD")
                    print(difc_LCRC_psd)
                    f, difg_o1_psd = signal.welch(difg_o1, nperseg=psd_size)
                    f, difc_o1_psd = signal.welch(difc_o1, nperseg=psd_size)
                    f, difg_o2_psd = signal.welch(difg_o2, nperseg=psd_size)
                    f, difc_o2_psd = signal.welch(difc_o2, nperseg=psd_size)

                    # cross power spectral density
                    left_C.fillna(0, inplace=True)
                    center_C.fillna(0, inplace=True)
                    right_C.fillna(0, inplace=True)
                    outerL_C.fillna(0, inplace=True)
                    outerR_C.fillna(0, inplace=True)

                    f, difc_LCCe_v_csd = signal.csd(left_C, center_C, nperseg=128)
                    f, difc_LCRC_v_csd = signal.csd(left_C, right_C, nperseg=128)
                    f, difc_RCCe_v_csd = signal.csd(right_C, center_C, nperseg=128)
                    f, difc_LCOL_v_csd = signal.csd(left_C, outerL_C, nperseg=128)
                    f, difc_RCOR_v_csd =signal.csd(right_C, outerR_C, nperseg=128)

                    difc_LCCe_csd_0 = []
                    difc_LCRC_csd_0 = []
                    difc_RCCe_csd_0 = []
                    difc_LCOL_csd_0 = []
                    difc_RCOR_csd_0 = []
                    difc_LCCe_csd_1 = []
                    difc_LCRC_csd_1 = []
                    difc_RCCe_csd_1 = []
                    difc_LCOL_csd_1 = []
                    difc_RCOR_csd_1 = []

                    for i in range(len(difc_LCCe_v_csd)):
                        difc_LCCe_csd_0.append(difc_LCCe_v_csd[i].real)
                        difc_LCCe_csd_1.append(difc_LCCe_v_csd[i].imag)

                    for i in range(len(difc_LCRC_v_csd)):
                        difc_LCRC_csd_0.append(difc_LCRC_v_csd[i].real)
                        difc_LCRC_csd_1.append(difc_LCRC_v_csd[i].imag)

                    for i in range(len(difc_RCCe_v_csd)):
                        difc_RCCe_csd_0.append(difc_RCCe_v_csd[i].real)
                        difc_RCCe_csd_1.append(difc_RCCe_v_csd[i].imag)

                    for i in range(len(difc_LCOL_v_csd)):
                        difc_LCOL_csd_0.append(difc_LCOL_v_csd[i].real)
                        difc_LCOL_csd_1.append(difc_LCOL_v_csd[i].imag)

                    for i in range(len(difc_RCOR_v_csd)):
                        difc_RCOR_csd_0.append(difc_RCOR_v_csd[i].real)
                        difc_RCOR_csd_1.append(difc_RCOR_v_csd[i].imag)

                    csd2_LCCe = []
                    csd2_LCRC = []
                    csd2_RCCe = []

                    for i in range(len(difc_RCCe_csd_0)):
                        csd2_LCCe.append((difc_LCCe_csd_0[i], difc_LCCe_csd_1[i]))
                        csd2_LCRC.append((difc_LCRC_csd_0[i], difc_LCRC_csd_1[i]))
                        csd2_RCCe.append((difc_RCCe_csd_0[i], difc_RCCe_csd_1[i]))

                    # f1 feature


                    t = np.abs(difc_LCCe_v_csd)
                    j = np.argmax(t)

                    max_cLCCe = (difc_LCCe_csd_0[j], difc_LCCe_csd_1[j])
                    mean_cLCCe = [np.mean(np.asarray(difc_LCCe_csd_0)), np.mean(np.asarray(difc_LCCe_csd_1))]

                    f1LCCe = np.array([max_cLCCe[0], max_cLCCe[1], mean_cLCCe[0], mean_cLCCe[1]])

                    t = np.abs(difc_LCRC_v_csd)
                    j = np.argmax(t)

                    max_cLCRC = (difc_LCRC_csd_0[j], difc_LCRC_csd_1[j])
                    mean_cLCRC = [np.mean(np.asarray(difc_LCRC_csd_0)), np.mean(np.asarray(difc_LCRC_csd_1))]

                    f1LCRC = np.array([max_cLCRC[0], max_cLCRC[1], mean_cLCRC[0], mean_cLCRC[1]])

                    t = np.abs(difc_RCCe_v_csd)
                    j = np.argmax(t)

                    max_cRCCe = (difc_RCCe_csd_0[j], difc_RCCe_csd_1[j])
                    mean_cRCCe = [np.mean(np.asarray(difc_RCCe_csd_0)), np.mean(np.asarray(difc_RCCe_csd_1))]

                    f1RCCe = np.array([max_cRCCe[0], max_cRCCe[1], mean_cRCCe[0], mean_cRCCe[1]])

                    t = np.abs(difc_LCOL_v_csd)
                    j = np.argmax(t)

                    max_cLCOL = (difc_LCOL_csd_0[j], difc_LCOL_csd_1[j])
                    mean_cLCOL = [np.mean(np.asarray(difc_LCOL_csd_0)), np.mean(np.asarray(difc_LCOL_csd_1))]

                    f1LCOL = np.array([max_cLCOL[0], max_cLCOL[1], mean_cLCOL[0], mean_cLCOL[1]])

                    t = np.abs(difc_RCOR_v_csd)
                    j = np.argmax(t)

                    max_cRCOR = (difc_RCOR_csd_0[j], difc_RCOR_csd_1[j])
                    mean_cRCOR = [np.mean(np.asarray(difc_RCOR_csd_0)), np.mean(np.asarray(difc_RCOR_csd_1))]

                    f1RCOR = np.array([max_cRCOR[0], max_cRCOR[1], mean_cRCOR[0], mean_cRCOR[1]])

                    derived_data_mean = np.concatenate([difg_LCRC_mean, difc_LCRC_mean, difg_o1_mean, difc_o1_mean,
                                                       difg_o2_mean, difc_o2_mean])

                    derived_data_var = np.concatenate([difg_LCRC_var, difc_LCRC_var, difg_o1_var, difc_o1_var,
                                                       difg_o2_var, difc_o2_var])

                    derived_data_std = np.concatenate([difg_LCRC_std, difc_LCRC_std, difg_o1_std, difc_o1_std,
                                                        difg_o2_std, difc_o2_std])

                    derived_data_psd = np.concatenate([difc_LCCe_psd, difc_LCRC_psd, difc_RCCe_psd])

                    derived_data_csd = np.concatenate([difc_LCCe_csd_0, difc_LCCe_csd_1, difc_LCRC_csd_0, difc_LCRC_csd_1, difc_RCCe_csd_0, difc_RCCe_csd_1])

                    derived_data_rcsd = np.concatenate([difc_LCCe_csd_0, difc_LCRC_csd_0, difc_RCCe_csd_0])

                    derived_data_f1 = np.concatenate([f1LCCe, f1LCRC, f1RCCe])

                    derived_data_test = np.concatenate([f1LCCe, f1LCRC, f1RCCe, f1LCOL, f1RCOR, difc_LCRC_std, difc_LCRC_var, difc_LCRC_psd, difc_LCRC_mean])

                    chrom_data = self.dataset['RC-chrom'].iloc[50] - self.dataset['C-chrom'].iloc[50]


                    if 'fake' in self.full_path.lower():
                        self.fake_data_LCCE[self.fake_count] = difc_LCCe
                        self.fake_data_LCRC[self.fake_count] = difc_LCRC
                        self.fake_data_RCCE[self.fake_count] = difc_RCCe
                        self.fake_data_mean[self.fake_count] = derived_data_mean
                        self.fake_data_var[self.fake_count] = derived_data_var
                        self.fake_data_std[self.fake_count] = derived_data_std
                        self.fake_data_psd[self.fake_count] = derived_data_psd
                        self.fake_data_csd[self.fake_count] = derived_data_csd
                        self.fake_data_f1[self.fake_count] = derived_data_f1
                        self.fake_data_test[self.fake_count] = derived_data_test
                        self.fake_count += 1
                    else:
                        self.real_data_LCCE[self.real_count] = difc_LCCe
                        self.real_data_LCRC[self.real_count] = difc_LCRC
                        self.real_data_RCCE[self.real_count] = difc_RCCe
                        self.real_data_mean[self.real_count] = derived_data_mean
                        self.real_data_var[self.real_count] = derived_data_var
                        self.real_data_std[self.real_count] = derived_data_std
                        self.real_data_psd[self.real_count] = derived_data_psd
                        self.real_data_csd[self.real_count] = derived_data_csd
                        self.real_data_f1[self.real_count] = derived_data_f1
                        self.real_data_test[self.real_count] = derived_data_test
                        self.real_count += 1

                    self.vid_count += 1

        self.real_df_LCCE = pd.DataFrame(self.real_data_LCCE)
        self.real_df_LCRC = pd.DataFrame(self.real_data_LCRC)
        self.real_df_RCCE = pd.DataFrame(self.real_data_RCCE)
        self.fake_df_LCCE = pd.DataFrame(self.fake_data_LCCE)
        self.fake_df_LCRC = pd.DataFrame(self.fake_data_LCRC)
        self.fake_df_RCCE = pd.DataFrame(self.fake_data_RCCE)
        self.real_df_m = pd.DataFrame(self.real_data_mean)
        self.fake_df_m = pd.DataFrame(self.fake_data_mean)
        self.real_df_v = pd.DataFrame(self.real_data_var)
        self.fake_df_v = pd.DataFrame(self.fake_data_var)
        self.real_df_s = pd.DataFrame(self.real_data_std)
        self.fake_df_s = pd.DataFrame(self.fake_data_std)
        self.real_df_p = pd.DataFrame(self.real_data_psd)
        self.fake_df_p = pd.DataFrame(self.fake_data_psd)
        self.real_df_csp = pd.DataFrame(self.real_data_csd)
        self.fake_df_csp = pd.DataFrame(self.fake_data_csd)
        self.real_df_f1 = pd.DataFrame(self.real_data_f1)
        self.fake_df_f1 = pd.DataFrame(self.fake_data_f1)
        self.real_df_test = pd.DataFrame(self.real_data_test)
        self.fake_df_test = pd.DataFrame(self.fake_data_test)
        r_lcce = self.real_df_LCCE.transpose()
        r_lcrc = self.real_df_LCRC.transpose()
        r_rcce = self.real_df_RCCE.transpose()
        f_lcce = self.fake_df_LCCE.transpose()
        f_lcrc = self.fake_df_LCRC.transpose()
        f_rcce = self.fake_df_RCCE.transpose()
        r_m = self.real_df_m.transpose()
        f_m = self.fake_df_m.transpose()
        r_v = self.real_df_v.transpose()
        f_v = self.fake_df_v.transpose()
        r_s = self.real_df_s.transpose()
        f_s = self.fake_df_s.transpose()
        r_p = self.real_df_s.transpose()
        f_p = self.fake_df_s.transpose()
        r_c = self.real_df_csp.transpose()
        f_c = self.fake_df_csp.transpose()
        r_f = self.real_df_f1.transpose()
        f_f = self.fake_df_f1.transpose()
        r_t = self.real_df_test.transpose()
        f_t = self.fake_df_test.transpose()
        r_f.to_csv("./real_f1.csv", index=False)
        f_f.to_csv("./fake_f1.csv", index=False)
        r_lcce['Target'] = 1
        r_lcrc['Target'] = 1
        r_rcce['Target'] = 1
        f_lcce['Target'] = 0
        f_lcrc['Target'] = 0
        f_rcce['Target'] = 0
        r_m['Target'] = 1
        f_m['Target'] = 0
        r_v['Target'] = 1
        f_v['Target'] = 0
        r_s['Target'] = 1
        f_s['Target'] = 0
        r_p['Target'] = 1
        f_p['Target'] = 0
        r_c['Target'] = 1
        f_c['Target'] = 0
        r_f['Target'] = 1
        f_f['Target'] = 0
        r_t['Target'] = 1
        f_t['Target'] = 0
        rf_lcce = r_lcce.append(f_lcce)
        rf_lcrc = r_lcrc.append(f_lcrc)
        rf_rcce = r_rcce.append(f_rcce)
        rf_m = r_m.append(f_m)
        rf_v = r_v.append(f_v)
        rf_s = r_s.append(f_s)
        rf_p = r_p.append(f_p)
        rf_c = r_c.append(f_c)
        rf_f = r_f.append(f_f)
        rf_t = r_t.append(f_t)
        test_v, train_v = train_test_split(rf_v, test_size=0.2)
        test_m, train_m = train_test_split(rf_m, test_size=0.2)
        test_s, train_s = train_test_split(rf_s, test_size=0.2)
        test_p, train_p = train_test_split(rf_p, test_size=0.2)
        test_c, train_c = train_test_split(rf_c, test_size=0.2)
        test_f, train_f = train_test_split(rf_f, test_size=0.2)
        test_t, train_t = train_test_split(rf_t, test_size=0.2)
        test_lcce, train_lcce = train_test_split(rf_lcce, test_size=0.2)
        test_lcrc, train_lcrc = train_test_split(rf_lcrc, test_size=0.2)
        test_rcce, train_rcce = train_test_split(rf_rcce, test_size=0.2)
        train_lcce.to_csv(self.train_data_lcce_path, index=False)
        train_lcrc.to_csv(self.train_data_lcrc_path, index=False)
        train_rcce.to_csv(self.train_data_rcce_path, index=False)
        test_lcce.to_csv(self.test_data_lcce_path, index=False)
        test_lcrc.to_csv(self.test_data_lcrc_path, index=False)
        test_rcce.to_csv(self.test_data_rcce_path, index=False)
        train_s.to_csv(self.train_data_s_path, index=False)
        test_s.to_csv(self.test_data_s_path, index=False)
        train_v.to_csv(self.train_data_v_path, index=False)
        test_v.to_csv(self.test_data_v_path, index=False)
        train_m.to_csv(self.train_data_m_path, index=False)
        test_m.to_csv(self.test_data_m_path, index=False)
        train_p.to_csv(self.train_data_p_path, index=False)
        test_p.to_csv(self.test_data_p_path, index=False)
        train_c.to_csv(self.train_data_c_path, index=False)
        test_c.to_csv(self.test_data_c_path, index=False)
        train_f.to_csv(self.train_data_f1_path, index=False)
        test_f.to_csv(self.test_data_f1_path, index=False)
        train_t.to_csv(self.train_data_test_path, index=False)
        test_t.to_csv(self.test_data_test_path, index=False)
        r_c.to_csv("./csd_real128.csv", index=False)
        f_c.to_csv("./csd_fake128.csv", index=False)


p = ProcessSignalData()
