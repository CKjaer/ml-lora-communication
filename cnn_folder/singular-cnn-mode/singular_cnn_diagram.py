import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Add flow diagram boxes
boxes = {
    "Averaging Weights": (1, 4),
    "Initialize Singular Model": (4, 4),
    "Fine-Tune with Combined Dataset": (7, 4)
}
for text, pos in boxes.items():
    ax.text(pos[0], pos[1], text, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue"), fontsize=10)

# Add arrows
arrow_params = dict(arrowstyle='->', color='black', lw=1.5)
ax.annotate("", xy=(3, 4), xytext=(2, 4), arrowprops=arrow_params)  # From Averaging to Initialization
ax.annotate("", xy=(6, 4), xytext=(5, 4), arrowprops=arrow_params)  # From Initialization to Fine-Tune

# Title
plt.title("Flow Diagram: Singular CNN Model Creation", fontsize=14, weight='bold')
plt.show()
