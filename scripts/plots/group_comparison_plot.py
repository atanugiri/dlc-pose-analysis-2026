import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_group_comparison(*value_lists, labels=None, ax=None, capsize=5, ylabel="Mean ± SE", show_points=True, test='welch', plot_type='bar'):
	"""Plot grouped values as bar (mean +/- SE) or box plots.
	
	Args:
		test: 'welch' (two-tailed) or 'mann_whitney'
		plot_type: 'bar' or 'box'
	"""
	if len(value_lists) == 0:
		raise ValueError("Provide at least one list of values.")
	if plot_type not in {'bar', 'box'}:
		raise ValueError("plot_type must be 'bar' or 'box'.")
	if test not in {'welch', 'mann_whitney'}:
		raise ValueError("test must be 'welch' or 'mann_whitney'.")

	arrays = [np.asarray(v, dtype=float) for v in value_lists]
	arrays = [a[np.isfinite(a)] for a in arrays]
	if any(len(a) == 0 for a in arrays):
		raise ValueError("Each input list must contain at least one finite value.")
	means = [a.mean() for a in arrays]
	ses = [a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0 for a in arrays]

	def _welch_t_with_df(a1, a2, alternative='two-sided'):
		res = stats.ttest_ind(a1, a2, equal_var=False, alternative=alternative)
		if hasattr(res, "statistic") and hasattr(res, "pvalue"):
			stat = float(res.statistic)
			p_value = float(res.pvalue)
			df = getattr(res, "df", np.nan)
			if np.isfinite(df):
				return stat, p_value, float(df)
		else:
			stat, p_value = res

		n1, n2 = len(a1), len(a2)
		v1 = np.var(a1, ddof=1) if n1 > 1 else 0.0
		v2 = np.var(a2, ddof=1) if n2 > 1 else 0.0
		denom_sq = (v1 / n1 + v2 / n2) ** 2
		term1 = ((v1 / n1) ** 2) / (n1 - 1) if n1 > 1 else 0.0
		term2 = ((v2 / n2) ** 2) / (n2 - 1) if n2 > 1 else 0.0
		df_denom = term1 + term2
		df = denom_sq / df_denom if df_denom > 0 else np.nan
		return float(stat), float(p_value), float(df)

	stat_text = None
	if len(arrays) == 2:
		if test == 'mann_whitney':
			stat, p_value = stats.mannwhitneyu(arrays[0], arrays[1], alternative='two-sided')
			stat_text = f"Mann-Whitney U: U={stat:.3g}, p={p_value:.3g}"
		else:
			stat, p_value, df = _welch_t_with_df(arrays[0], arrays[1])
			stat_text = f"Welch t-test: t({df:.2f}) = {stat:.3g}, p={p_value:.3g}"
	elif len(arrays) > 2:
		stat, p_value = stats.f_oneway(*arrays)
		df_between = len(arrays) - 1
		df_within = sum(len(a) for a in arrays) - len(arrays)
		stat_text = f"1-way ANOVA: F({df_between}, {df_within})={stat:.3g}, p={p_value:.3g}"

	if labels is None:
		labels = [f"Group {i + 1}" for i in range(len(arrays))]
	if len(labels) != len(arrays):
		raise ValueError("labels length must match number of input lists.")

	cmap = plt.get_cmap("tab10")
	colors = [cmap(i % 10) for i in range(len(arrays))]

	if ax is None:
		_, ax = plt.subplots()

	x = np.arange(len(arrays))
	if plot_type == 'bar':
		ax.bar(x, means, yerr=ses, color=colors, capsize=capsize)
	else:
		ax.boxplot(
			arrays,
			positions=x,
			widths=0.5,
			patch_artist=True,
			showfliers=True,
			boxprops=dict(facecolor="white", edgecolor="black"),
			medianprops=dict(color="black", linewidth=1.5),
			whiskerprops=dict(color="black"),
			capprops=dict(color="black"),
		)
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

	if show_points:
		for i, a in enumerate(arrays):
			x_i = np.random.normal(i, 0.05, size=len(a))
			ax.scatter(x_i, a, color=colors[i], alpha=0.6, edgecolor='k', zorder=3)

	return ax
