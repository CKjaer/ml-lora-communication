import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd


dir=r"C:\Users\lukas\Desktop\AAU\EIT7\Project\git\ml-lora-communication\cnn_output\Test_20241114-130105\data".replace("\\", "/")
files=os.listdir(dir)
for i in range(len(files)):
    df_file=pd.read_csv(os.path.join(dir,files[i]), header=None)
    SERs=[[] for _ in range(2)]
    
    for sec in df_file[0]:
        snr,ser=sec.split(";")
        SERs[0].append(snr)
        SERs[1].append(ser)

    plt.plot(SERs[0], SERs[1])



plt.show()
