import matplotlib.pyplot as plt

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

fig.tight_layout()
fig.suptitle("Sharing X axis")

ax[0].set_title("#1")
ax[0].plot([1, 2, 3, 4], label="1")
ax[0].plot([4, 3, 2, 1], label="2")
ax[0].legend()

ax[1].set_title("#2")
ax[1].plot([4, 3, 2, 1], label="3")
ax[1].plot([1, 2, 3, 4], label="4")
ax[1].legend()

plt.show()


""" 
subplot.py:13: MatplotlibDeprecationWarning:
Auto-removal of overlapping axes is deprecated since 3.6 and will be removed two minor releases later;
explicitly call ax.remove() as needed.
  plt.subplot(2, 2, 1)
"""

# plt.figure(figsize=(8, 8))

# plt.title("Subplot Example #1")
# plt.subplot(2, 2, 1)
# plt.plot([1, 2, 3, 4], label="1")
# plt.plot([4, 3, 2, 1], label="2")

# plt.title("Subplot Example #2")
# plt.subplot(2, 2, 2)
# plt.plot([4, 3, 2, 1], label="3")
# plt.plot([1, 2, 3, 4], label="4")

# plt.title("Subplot Example #3")
# plt.subplot(2, 2, 3)
# plt.plot([1, 2, 3, 4], label="5")
# plt.plot([4, 3, 2, 1], label="6")

# plt.title("Subplot Example #4")
# plt.subplot(2, 2, 4)
# plt.plot([4, 3, 2, 1], label="7")
# plt.plot([1, 2, 3, 4], label="8")

# plt.legend()
# plt.show()
