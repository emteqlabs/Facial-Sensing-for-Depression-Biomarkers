import os
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator


def set_plot_fonts(default=12, title=14, labels=12, ticks=12, legend=12):
    """Customize plot font sizes"""
    plt.rc("font", size=default)  # controls default text sizes
    plt.rc("axes", titlesize=title)  # fontsize of the axes title
    plt.rc("axes", labelsize=labels)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=ticks)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=ticks)  # fontsize of the tick labels
    plt.rc("legend", fontsize=legend)  # legend fontsize
    plt.rc("figure", titlesize=title)  # fontsize of the figure title


def extract_test_results(test_results):
    """Extract results from statistical tests"""
    results = []
    for test in test_results:
        results.append(test.data.__dict__)
    return results


def plot_statistics(
    data,
    pairs,
    x,
    y,
    hue=None,
    test="Mann-Whitney",
    figsize=(12, 6),
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    path_to_save=None,
    show_plot=True,
    show_legend=True,
    palette=None,
    ax=None,
    bbox_to_anchor=None,
):
    """
    Create a boxplot with statistical annotations using seaborn and
    statannotations.

    This function visualizes group differences and applies a statistical test
    with Bonferroni correction to annotated pairs. It supports customization of
    plot appearance and optionally saves the figure to disk.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to plot.
    pairs : list of tuple
        List of tuples specifying the group pairs to compare
        (e.g., [("group1", "group2")]).
    x : str
        Name of the column to be used on the x-axis (grouping variable).
    y : str
        Name of the column to be used on the y-axis (value variable).
    hue : str, optional
        Column name for additional grouping
        (e.g., categories within each x group).
    test : str, default "Mann-Whitney"
        The statistical test to apply to the specified pairs.
    figsize : tuple, default (12, 6)
        Size of the plot figure.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Custom label for the x-axis.
    ylabel : str, optional
        Custom label for the y-axis.
    xlim : tuple, optional
        Limits for the x-axis (e.g., (0, 10)).
    ylim : tuple, optional
        Limits for the y-axis (e.g., (0, 1)).
    path_to_save : str, optional
        Directory path to save the plot. If provided, the figure is
        saved as a PNG.
    show_plot : bool, default True
        Whether to display the plot after creation.
    show_legend : bool, default True
        Whether to display the legend.
    palette : dict or str, optional
        Color palette for the plot
        (e.g., 'Set2' or {'label1': 'blue', 'label2': 'red'}).
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib Axes object to draw the plot on. If None, a
        new figure is created.
    bbox_to_anchor : tuple, optional
        Anchor position for the legend when shown.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the results of the statistical tests for each pair,
        including test statistics and p-values.
    """
    plotting_parameters = {
        "x": x,
        "y": y,
        "data": data,
        "hue": hue,
    }
    if palette is not None:
        plotting_parameters["palette"] = palette
    created_ax = False
    # Use the provided ax or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_ax = True
    else:
        fig = ax.figure  # Get the figure from the existing axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    sns.boxplot(ax=ax, **plotting_parameters)
    annotator = Annotator(ax=ax, pairs=pairs, **plotting_parameters)
    _, test_results = annotator.configure(
        test=test, comparisons_correction="Bonferroni"
    ).apply_and_annotate()
    if hue is not None:
        if show_legend:
            if bbox_to_anchor is not None:
                ax.legend(loc="upper left", bbox_to_anchor=bbox_to_anchor)
            else:
                ax.legend(loc="upper right")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        fig.suptitle(title, fontweight="bold")
    # if not show_legend:
    #     ax.get_legend().remove()
    plt.tight_layout()

    if show_plot:
        plt.show()
    # else:
    #     plt.close(fig)

    if path_to_save is not None:
        os.makedirs(path_to_save, exist_ok=True)
        if title is not None:
            filename = f"{title}.png"
        else:
            filename = f"{test}_boxplots.png"
        fig.savefig(os.path.join(path_to_save, filename))
    if created_ax:
        if not sys.stdin.isatty():
            plt.show()
        elif path_to_save is not None:
            plt.close(fig)
        else:
            plt.show()
        plt.rcdefaults()
    return extract_test_results(test_results)


def print_results(results):
    """
    Print formatted statistical test results for a list of pairwise comparisons.

    This function displays each result as a markdown-style table using the
    `tabulate` library. Each comparison is titled using the compared
    group names.

    Parameters
    ----------
    results : list of dict
        A list of dictionaries where each dictionary contains the results of a
        statistical test. Each dictionary must include the keys 'group1'
        and 'group2' (both iterable) along with other key-value pairs
        representing test statistics (e.g., p-value, effect size).

    Returns
    -------
    None
        This function prints output to the console and does not return anything.
    """
    for result in results:
        rows = [(k, v) for k, v in result.items()]
        title = (
            f"{', '.join(result['group1'])} vs {', '.join(result['group2'])}"
        )
        print(f"\n**{title}**\n")
        print(tabulate(rows, headers=["Field", "Value"], tablefmt="github"))
