import numpy as np
import matplotlib.pyplot as plt


def barplot_mean_se(*value_lists, labels=None, colors=None, ax=None, capsize=5):
	"""Plot mean bars with standard error for variable list inputs."""
	if len(value_lists) == 0:
		raise ValueError("Provide at least one list of values.")

	arrays = [np.asarray(v, dtype=float) for v in value_lists]
	means = [a.mean() for a in arrays]
	ses = [a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0 for a in arrays]

	if labels is None:
		labels = [f"Group {i + 1}" for i in range(len(arrays))]
	if len(labels) != len(arrays):
		raise ValueError("labels length must match number of input lists.")

	if colors is None:
		cmap = plt.get_cmap("tab10")
		colors = [cmap(i % 10) for i in range(len(arrays))]
	if len(colors) != len(arrays):
		raise ValueError("colors length must match number of input lists.")

	if ax is None:
		_, ax = plt.subplots()

	x = np.arange(len(arrays))
	ax.bar(x, means, yerr=ses, color=colors, capsize=capsize)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_ylabel("Mean ± SE")

	return ax

