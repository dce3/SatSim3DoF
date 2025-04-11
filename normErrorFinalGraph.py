import numpy as np
import matplotlib.pyplot as plt

# Data arrays.
errors = np.array([1810406964.024427, 1466599648.749428, 1087736879.868660, 629881925.457010, 2333162.748120, 2333146.460086])
hs = np.array([16, 8, 4, 2, 1, 0.5])

plt.figure(figsize=(10, 6))

# Plot the error vs. step size.
plt.plot(hs, errors, "o-", lw=2, markersize=5, label="Final Positional Error")

# Add labels to each data point.
for h, err in zip(hs, errors):
    # Format error in scientific notation.
    label = "h="+str(h)+"s"
    plt.text(h, err+10000000, label, fontsize=10, ha="center", va="bottom", rotation=90)

# Label axes and title.
plt.xlabel("Step Size: h [s]", fontsize=12)
plt.ylabel("Final Positional Error [m]", fontsize=12)
#plt.title("RK4 Integration: Final Positional Error vs. Step Size", fontsize=14)

# Add grid and legend.
plt.grid(True, linestyle="--", alpha=0.7)
#plt.legend(fontsize=12)
plt.tight_layout()

plt.show()
