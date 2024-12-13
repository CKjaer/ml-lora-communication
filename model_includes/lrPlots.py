import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("lr_vs_acc.csv")
data.drop(columns=["_wandb"], inplace=True)

# rename rows in name column
new_row_names = {row: " ".join(row.split("-")[1:3]).capitalize() for row in data["Name"]}
data["Name"] = data["Name"].replace(new_row_names)

# rename columns
data.rename(columns={"final_accuracy": "Accuracy", "lr": "Learning Rate"}, inplace=True)
data["Sweep"] = data["Name"].astype("category")

# fit linear regression
X = data["Learning Rate"].values.reshape(-1, 1)
y = data["Accuracy"].values
linReg = LinearRegression()
linReg.fit(X, y)

# setup plot params
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Palatino Linotype"
plt.rcParams["font.family"] = "Palatino Linotype"
fs = 20
plt.rcParams.update({"font.size": fs})

# plot the data
plt.figure(figsize=(8, 6))
plt.scatter(data["Learning Rate"], data["Accuracy"], c=data["Sweep"].cat.codes, cmap="viridis")
plt.plot(data["Learning Rate"], linReg.predict(X), color="black", linewidth=2)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0.004, 0.046, 0.01))
plt.grid()
plt.savefig("lr_vs_acc.pdf", format="pdf", bbox_inches="tight")
plt.show()
