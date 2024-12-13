import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("loss_data.csv")

# Keep only the columns ending with '- Loss' and exclude those ending with '- Loss_MIN'
loss_columns = [col for col in data.columns if col.endswith("- Loss") and not col.endswith("- Loss_MIN")]
filtered_data = data[loss_columns]


# rename columns
filtered_data.columns = [col.replace("- Loss", "") for col in filtered_data.columns]
new_col_names = {col: " ".join(col.split("-")[1:3]).capitalize() for col in filtered_data.columns}
filtered_data.rename(columns=new_col_names, inplace=True)

# add epoch column
filtered_data["Epoch"] = np.arange(1, len(filtered_data) + 1)


# sample 5 random columns
sweeps = filtered_data.sample(axis="columns", n=8)
print(sweeps.columns)

# plot the sampled sweeps
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Palatino Linotype"
plt.rcParams["font.family"] = "Palatino Linotype"
fs = 20
plt.rcParams.update({"font.size": fs})


plt.figure(figsize=(8, 6))
for col in sweeps.columns:
    plt.plot(filtered_data["Epoch"], filtered_data[col], label=col)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("loss_vs_epoch.pdf", format="pdf", bbox_inches="tight")
plt.show()
