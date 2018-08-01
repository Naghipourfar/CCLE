from cmapPy.pandasGEXpress.GCToo import GCToo
from cmapPy.pandasGEXpress.parse import parse
import numpy as np
import pandas as pd


DATA_PATH = "/Users/Future/Desktop/Summer 2018/Bioinformatics/CCLE/Data/CCLE_DepMap_18Q2_RNAseq_reads_20180502.gct"

data = parse(DATA_PATH)
print(data, type(data))

print(data.data_df)
print(type(data.data_df))



