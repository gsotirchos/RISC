import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    cmap="viridis",
    mask=None,
    vmin=None,
    vmax=None,
    linewidths=1.5,
    square=True,
    logscale=False,
    **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()
    norm = (
        LogNorm(vmin=vmin, vmax=vmax) if logscale else Normalize(vmax=vmax, vmin=vmin)
    )
    ax = sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        norm=norm,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        square=square,
        **kwargs,
    )

    return ax
