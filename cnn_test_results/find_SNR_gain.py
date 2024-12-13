import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

filepath = os.path.abspath(__file__)
# print(filepath)
directory = os.path.abspath(os.path.join(filepath, "../snr_sims"))
# print(directory)
test_time = "2024_11_27_13_22_04"
# Initialize data_list as a list of dictionaries
# SF, SNR, error count, simulated symbols, SER
data_list = []
for filename in os.listdir(directory):
    if filename.startswith(test_time) & filename.endswith('.csv'):
        fp = os.path.join(directory, filename)
        rate = filename.split('_')[-1].removeprefix('lam').removesuffix('.csv')
        with open(fp) as f:
            lines = f.readlines()
            for line in lines:
                sep = line.strip().split(';')
                new_row = {'Rate': float(rate), 'SNR': sep[0], 'SER': sep[1]}
                data_list.append(new_row)
            f.close()
df_classical = pd.DataFrame(data_list)


names=["exp_scaled","batch_scaled","auto_scaled"]
df=[]
current_dir=os.path.dirname(os.path.realpath(__file__))
for name in names:
    test_type= name
    df.append(pd.read_csv(os.path.join(current_dir, "test_data",f"test_{test_type}.csv")))
rate_params=df[0].columns[1:]

# Prepare data for interpolation
ser_target = 1e-1
snr_targets = {}
for i, rate_param in enumerate(rate_params):
    classical=df_classical[df_classical["Rate"]==float(rate_param)]

    df_classical1 = interp1d(np.array(classical["SER"]).astype(float), np.array(classical["SNR"]).astype(float), kind='linear', fill_value="extrapolate")
    # df_classical1=interp1d(df_classical["SNR"]==rate_param, df_classical["SER"], kind='linear', fill_value="extrapolate")
    interpolators1 = interp1d(df[0].iloc[:, i], df[0].iloc[:, 0], kind='linear', fill_value="extrapolate")
    interpolators2 = interp1d(df[1].iloc[:, i], df[1].iloc[:, 0], kind='linear', fill_value="extrapolate")
    interpolators3 = interp1d(df[2].iloc[:, i], df[2].iloc[:, 0], kind='linear', fill_value="extrapolate")
    x=df_classical1(ser_target)
    print(x)
    snr_targets[f"{rate_param}_exp_scaled"] = interpolators1(ser_target)-x
    snr_targets[f"{rate_param}_batch_scaled"] = interpolators2(ser_target)-x
    snr_targets[f"{rate_param}_auto_scaled"] = interpolators3(ser_target)-x
    

# Perform interpolation for each rate parameter
# for i, rate_param in enumerate(rate_params):
#     interpolator = interpolators1[i]
#     snr_targets[rate_param] = interpolator(ser_target)

print(snr_targets)

# Reverse interpolation to find the SNR value for a target SER


# rate=0.0
# current_data = df_classical[df_classical["Rate"]==rate]
# interpolator1 = interp1d(np.array(current_data["SER"]).astype(float), np.array(current_data["SNR"]).astype(float), kind='linear')
# interpolator1 = interp1d(list(df[0].iloc[:,0])[::-1], list(df[0].iloc[:,1])[::-1], kind='linear')
# snr_target1 = interpolator1(ser_target)
# print(snr_target1)

# interpolator1 = interp1d(np.array(df_classical["SNR"]).astype(float), np.array(current_data["SER"]).astype(float), kind='linear')
# print(np.array(current_data["SNR"]).astype(float), np.array(current_data["SER"]).astype(float))
# interpolator2 = interp1d(df[x].iloc[:,0], df[x].iloc[:,i+1], kind='linear')

# snr_target1 = interpolator1(ser_target)
# snr_target2 = interpolator2(ser_target)
# print(snr_target1, snr_target2)
