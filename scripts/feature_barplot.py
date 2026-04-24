import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def barplot_mean_se(*value_lists, labels=None, colors=None, ax=None, capsize=5, ylabel="Mean ± SE"):
	"""Plot mean bars with standard error for variable list inputs."""
	if len(value_lists) == 0:
		raise ValueError("Provide at least one list of values.")

	arrays = [np.asarray(v, dtype=float) for v in value_lists]
	arrays = [a[np.isfinite(a)] for a in arrays]
	if any(len(a) == 0 for a in arrays):
		raise ValueError("Each input list must contain at least one finite value.")
	means = [a.mean() for a in arrays]
	ses = [a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0 for a in arrays]
	stat_text = None
	if len(arrays) == 2:
		stat, p_value = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
		stat_text = f"Welch t-test: t={stat:.3g}, p={p_value:.3g}"
	elif len(arrays) > 2:
		stat, p_value = stats.f_oneway(*arrays)
		df_between = len(arrays) - 1
		df_within = sum(len(a) for a in arrays) - len(arrays)
		stat_text = f"1-way ANOVA: F({df_between}, {df_within})={stat:.3g}, p={p_value:.3g}"

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
	ax.set_ylabel(ylabel)
	if stat_text is not None:
		ax.text(
			0.5,
			0.98,
			stat_text,
			transform=ax.transAxes,
			ha="center",
			va="top",
			bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
		)

	return ax

