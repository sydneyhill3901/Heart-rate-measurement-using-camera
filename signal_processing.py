import os, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import signal

class ProcessSignalData(object):
    def __init__(self):
        self.dir = './processed/videos'
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
        self.real_count = 0
        self.fake_count = 0
        self.vid_count = 0
        self.data_path_m = './mean_data16.csv'
        self.data_path_v = './var_data16.csv'
        self.data_path_s = './std_data16.csv'
        self.data_path_p = './psd_data16.csv'
        self.log_path = './process_log.csv'
        self.test_data_v_path = './train_data_v16.csv'
        self.train_data_v_path = './test_data_v16.csv'
        self.test_data_m_path = './train_data_m16.csv'
        self.train_data_m_path = './test_data_m16.csv'
        self.test_data_s_path = './train_data_s16.csv'
        self.train_data_s_path = './test_data_s16.csv'
        self.test_data_p_path = './train_data_p64.csv'
        self.train_data_p_path = './test_data_p64.csv'

        self.main()

    def main(self):

        # length of video in frames to process
        sample_length = 250

        # interval for mean, var, std
        group_size = 16

        #window for psd
        psd_size = 64

        for paths, subdir, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.csv'):
                    self.full_path = os.path.join(paths, file)
                if 'rejected' in self.full_path.lower() or '.txt' in self.full_path.lower() or 'imposter' in self.full_path.lower():
                    pass
                else:
                    print(self.full_path)
                    self.dataset = pd.read_csv(self.full_path)
                    difg_LCRC = (self.dataset['RC-G'].iloc[:sample_length] - self.dataset['LC-G'].iloc[:sample_length] ).abs()
                    difc_LCRC = (self.dataset['RC-C'].iloc[:sample_length] - self.dataset['LC-C'].iloc[:sample_length] ).abs()
                    difg_o1 = (self.dataset['C-G'].iloc[:sample_length] - self.dataset['F-G'].iloc[:sample_length] ).abs()
                    difc_o1 = (self.dataset['C-C'].iloc[:sample_length] - self.dataset['F-C'].iloc[:sample_length] ).abs()
                    difg_o2 = (self.dataset['OR-G'].iloc[:sample_length] - self.dataset['OL-G'].iloc[:sample_length] ).abs()
                    difc_o2 = (self.dataset['OR-C'].iloc[:sample_length] - self.dataset['OL-C'].iloc[:sample_length] ).abs()


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
                    difg_o1_mean = np.array([difg_o1_lst[i].mean() for i in range(len(difg_o1_lst))])
                    difc_o1_mean = np.array([difc_o1_lst[i].mean() for i in range(len(difc_o1_lst))])
                    difg_o2_mean = np.array([difg_o2_lst[i].mean() for i in range(len(difg_o2_lst))])
                    difc_o2_mean = np.array([difc_o2_lst[i].mean() for i in range(len(difc_o2_lst))])

                    # variance
                    difg_LCRC_var = np.array([difg_LCRC_lst[i].var() for i in range(len(difg_LCRC_lst))])
                    difc_LCRC_var = np.array([difc_LCRC_lst[i].var() for i in range(len(difc_LCRC_lst))])
                    difg_o1_var = np.array([difg_o1_lst[i].var() for i in range(len(difg_o1_lst))])
                    difc_o1_var = np.array([difc_o1_lst[i].var() for i in range(len(difc_o1_lst))])
                    difg_o2_var = np.array([difg_o2_lst[i].var() for i in range(len(difg_o2_lst))])
                    difc_o2_var = np.array([difc_o2_lst[i].var() for i in range(len(difc_o2_lst))])

                    # standard deviation
                    difg_LCRC_std = np.array([difg_LCRC_lst[i].std() for i in range(len(difg_LCRC_lst))])
                    difc_LCRC_std = np.array([difc_LCRC_lst[i].std() for i in range(len(difc_LCRC_lst))])
                    difg_o1_std = np.array([difg_o1_lst[i].std() for i in range(len(difg_o1_lst))])
                    difc_o1_std = np.array([difc_o1_lst[i].std() for i in range(len(difc_o1_lst))])
                    difg_o2_std = np.array([difg_o2_lst[i].std() for i in range(len(difg_o2_lst))])
                    difc_o2_std = np.array([difc_o2_lst[i].std() for i in range(len(difc_o2_lst))])

                    # power spectral density
                    f, difg_LCRC_psd = signal.welch(difg_LCRC, nperseg=psd_size)
                    f, difc_LCRC_psd = signal.welch(difc_LCRC, nperseg=psd_size)
                    f, difg_o1_psd = signal.welch(difg_o1, nperseg=psd_size)
                    f, difc_o1_psd = signal.welch(difc_o1, nperseg=psd_size)
                    f, difg_o2_psd = signal.welch(difg_o2, nperseg=psd_size)
                    f, difc_o2_psd = signal.welch(difc_o2, nperseg=psd_size)



                    """
                    derived_data_mean = pd.DataFrame({'LCRC-G-mean': difg_LCRC_mean,
                                                      'LCRC-C-mean': difc_LCRC_mean,
                                                      'CF-G-mean': difg_o1_mean,
                                                      'CF-C-mean': difc_o1_mean,
                                                      'O-G-mean': difg_o2_mean,
                                                      'O-C-mean': difc_o2_mean})

                    derived_data_var = pd.DataFrame({'LCRC-G-var': difg_LCRC_var,
                                                      'LCRC-C-var': difc_LCRC_var,
                                                      'CF-G-var': difg_o1_var,
                                                      'CF-C-var': difc_o1_var,
                                                      'O-G-var': difg_o2_var,
                                                      'O-C-var': difc_o2_var})

                    derived_data_std = pd.DataFrame({'LCRC-G-std': difg_LCRC_std,
                                                      'LCRC-C-std': difc_LCRC_std,
                                                      'CF-G-std': difg_o1_std,
                                                      'CF-C-std': difc_o1_std,
                                                      'O-G-std': difg_o2_std,
                                                      'O-C-std': difc_o2_std})

                    derived_data_psd = pd.DataFrame({'LCRC-G-psd': difg_LCRC_psd,
                                                      'LCRC-C-std': difc_LCRC_psd,
                                                      'CF-G-std': difg_o1_psd,
                                                      'CF-C-std': difc_o1_psd,
                                                      'O-G-std': difg_o2_psd,
                                                      'O-C-std': difc_o2_psd})

                    """

                    derived_data_mean = np.concatenate([difg_LCRC_mean, difc_LCRC_mean, difg_o1_mean, difc_o1_mean,
                                                       difg_o2_mean, difc_o2_mean])

                    derived_data_var = np.concatenate([difg_LCRC_var, difc_LCRC_var, difg_o1_var, difc_o1_var,
                                                       difg_o2_var, difc_o2_var])

                    derived_data_std = np.concatenate([difg_LCRC_std, difc_LCRC_std, difg_o1_std, difc_o1_std,
                                                        difg_o2_std, difc_o2_std])

                    derived_data_psd = np.concatenate([difg_LCRC_psd, difc_LCRC_psd, difg_o1_psd, difc_o1_psd,
                                                        difg_o2_psd, difc_o2_psd])

                    if 'fake' in self.full_path.lower():
                        self.fake_data_mean[self.fake_count] = derived_data_mean
                        self.fake_data_var[self.fake_count] = derived_data_var
                        self.fake_data_std[self.fake_count] = derived_data_std
                        self.fake_data_psd[self.fake_count] = derived_data_psd
                        self.fake_count += 1
                    else:
                        self.real_data_mean[self.real_count] = derived_data_mean
                        self.real_data_var[self.real_count] = derived_data_var
                        self.real_data_std[self.real_count] = derived_data_std
                        self.real_data_psd[self.real_count] = derived_data_psd
                        self.real_count += 1

                    self.vid_count += 1

        self.real_df_m = pd.DataFrame(self.real_data_mean)
        self.fake_df_m = pd.DataFrame(self.fake_data_mean)
        self.real_df_v = pd.DataFrame(self.real_data_var)
        self.fake_df_v = pd.DataFrame(self.fake_data_var)
        self.real_df_s = pd.DataFrame(self.real_data_std)
        self.fake_df_s = pd.DataFrame(self.fake_data_std)
        self.real_df_p = pd.DataFrame(self.real_data_psd)
        self.fake_df_p = pd.DataFrame(self.fake_data_psd)
        r_m = self.real_df_m.transpose()
        f_m = self.fake_df_m.transpose()
        r_v = self.real_df_v.transpose()
        f_v = self.fake_df_v.transpose()
        r_s = self.real_df_s.transpose()
        f_s = self.fake_df_s.transpose()
        r_p = self.real_df_s.transpose()
        f_p = self.fake_df_s.transpose()
        r_m['Target'] = 1
        f_m['Target'] = 0
        r_v['Target'] = 1
        f_v['Target'] = 0
        r_s['Target'] = 1
        f_s['Target'] = 0
        r_p['Target'] = 1
        f_p['Target'] = 0
        rf_m = r_m.append(f_m)
        rf_v = r_v.append(f_v)
        rf_s = r_s.append(f_s)
        rf_p = r_p.append(f_p)

        rf_m.to_csv(self.data_path_m, index=False)
        rf_v.to_csv(self.data_path_v, index=False)
        rf_s.to_csv(self.data_path_s, index=False)
        rf_p.to_csv(self.data_path_p, index=False)
        test_v, train_v = train_test_split(rf_v, test_size=0.2)
        test_m, train_m = train_test_split(rf_m, test_size=0.2)
        test_s, train_s = train_test_split(rf_s, test_size=0.2)
        test_p, train_p = train_test_split(rf_p, test_size=0.2)
        train_s.to_csv(self.train_data_s_path, index=False)
        test_s.to_csv(self.test_data_s_path, index=False)
        train_v.to_csv(self.train_data_v_path, index=False)
        test_v.to_csv(self.test_data_v_path, index=False)
        train_m.to_csv(self.train_data_m_path, index=False)
        test_m.to_csv(self.test_data_m_path, index=False)
        train_p.to_csv(self.train_data_p_path, index=False)
        test_p.to_csv(self.test_data_p_path, index=False)


p = ProcessSignalData()
