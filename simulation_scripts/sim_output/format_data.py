import os
import numpy as np
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__))
type= "SIR"
dir=os.path.join(dir, "test_data",type)

out_list=os.listdir(dir)
out_list=[i for i in out_list if i.endswith(".log")]


progress= open(os.path.join(dir, "progress.txt"), "r")
data=open(os.path.join(dir, "data.txt"), "r")

data=data.readlines()
progress=progress.readlines()
log_data=[]
for log in out_list:
    file_out=open(os.path.join(dir, log), "r")
    file_out=file_out.readlines()
    for line in file_out:
        try:
            line1=line.split(":")
            if len(line1)==6:
                rate=line1[3].strip("[] rate")
                snr=line1[4].strip("[] Result SER")
                SER=line1[5].strip("\n")
                if [rate, snr, SER] not in log_data:
                    log_data.append([rate, snr, SER])
        except:
            continue

snr=[]
rate_params=[]
SER=[]
for line in log_data:
    snr.append(int(line[0]))
    rate_params.append(float(line[1]))
    SER.append(float(line[2]))

unique_snr=[]
unique_rate=[]
[unique_snr.append(i) for i in snr if i not in unique_snr]
[unique_rate.append(i) for i in rate_params if i not in unique_rate]
unique_snr=np.sort(unique_snr)
unique_rate= np.sort(unique_rate)
model_SERs=[[None] *len(unique_rate) for _ in range(len(unique_snr))]
for i in range(len(log_data)):
    model_SERs[np.where(unique_snr==snr[i])[0][0]][np.where(unique_rate==rate_params[i])[0][0]]=SER[i]


current_dir=os.path.dirname(os.path.realpath(__file__))
pd.DataFrame(model_SERs, columns=unique_rate, index=unique_snr).to_csv(os.path.join(current_dir,f"test_{type}.csv"))





