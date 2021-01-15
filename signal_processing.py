import os, csv
import numpy as np
import pandas as pd
from pathlib import Path

class ProcessSignalData(object):
    def __init__(self):
        self.dir = './processed/videos/test'
        self.full_path = ''
        self.real_output_path = self.dir + '/real_data.csv'
        self.fake_output_path = self.dir + '/fake_data.csv'
        self.real_data = pd.DataFrame()
        self.fake_data = pd.DataFrame()


    def main(self):

        for paths, subdir, files in os.walk(self.dir):
            for file in files:
                self.full_path = os.path.join(paths, file)
                if 'rejected' in self.full_path.lower():
                    pass
                else:
                    data = pd.read_csv(self.full_path)
                    green_data = pd.DataFrame(data, columns=['RC-G', 'LC-G', 'C-G', 'F-G', 'OR-G', 'OL-G'])
                    chrom_r_data = pd.DataFrame(data, columns=['RC-Cr', 'LC-Cr', 'C-Cr', 'F-Cr', 'OR-Cr', 'OL-Cr'])
                    chrom_g_data = pd.DataFrame(data, columns=['RC-Cg', 'LC-Cg', 'C-Cg', 'F-Cg', 'OR-Cg', 'OL-Cg'])
                    chrom_data = chrom_r_data - chrom_g_data
                    green_var = green_data.rolling(10).var(axis=1)
                    chrom_var = chrom_data.rolling(10).var(axis=1)
                    var = green_var.var(axis=0).join(chrom_var.var(axis=0)).transpose()
                    if 'real' in self.full_path.lower():
                        self.real_data.join(var)
                    else:
                        self.fake_data.join(var)

        self.real_data.to_csv(self.real_output_path, index=False)
        self.fake_data.to_csv(self.fake_output_path, index=False)




