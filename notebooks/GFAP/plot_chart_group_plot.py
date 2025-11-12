import pandas as pd 
import altair as alt 

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def figure_plot(df, title):
    fig = go.Figure()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]
    sample_sizes = [18, 24, 32, 36, 48, 64, 96, 112, 136]
    exp_types = ["Crossover design", "Parallel design"]
    
    config_pairs = [
        ("SEED design",      18, "#1f77b4"),
        ("Parallel design",  36, "#1f77b4"),
        ("SEED design",      24, "#ff7f0e"),
        ("Parallel design",  48, "#ff7f0e"),
        ("SEED design",      32, "#2ca02c"),
        ("Parallel design",  64, "#2ca02c"),
        ("SEED design",      48, "#7f7f7f"),
        ("Parallel design",  96, "#7f7f7f"),
        ("Parallel design", 112, "#8c564b"),
        ("Parallel design", 130, "#e377c2"),
        ("Parallel design", 136, "#17becf"),
    ]
    
    
    for exp_type, sample_size, color in config_pairs:
        data = df[(df["sample_size"] == sample_size) & (df["Experiment type"] == exp_type)]
        if data.empty:
            continue
    
        linestyle = "dash" if exp_type == "Parallel design" else None
        label = f"{exp_type} ({sample_size})"  # Combine design and sample size
        if linestyle:
            fig.add_trace(go.Scatter(x=data["Ratio"], y=data["Percentage"], name=label,
                                     line=dict(color=color, width=2,
                                          dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
            ))
        else:
            fig.add_trace(go.Scatter(x=data["Ratio"], y=data["Percentage"], name=label,
                                     line=dict(color=color, width=2) 
            ))
    fig.add_hline(y=0.8, line=dict(color="red", width=1))
    # Edit the layout
    fig.update_layout(
            title=dict(
                text=title
            ),
            xaxis=dict(
                title='\u03B2',
                tickmode = 'linear',
                tick0 = 0.0,
                dtick = 0.1,
            ),
            yaxis=dict(
                title='Power',
                tickformat='.0%',
                tick0 = 0.0,
                dtick = 0.1,
                range = [0,1.05]
            ),
            # width=1000,
            # height=500,
            showlegend=False,
    )
    return fig

def combine_plotly_charts(list_of_figures, n_cols=2, title="Temp", fig_width=None, fig_height=None, show_shared_legend=False):
    """
    Combines a list of Plotly figures into a single figure with a grid layout,
    and attempts to create a shared legend (minimizing duplicate entries).

    Args:
        list_of_figures: A list of Plotly Figure objects.
        n_cols: The number of columns in the subplot grid (default is 2).
        fig_width:  Width of the combined figure (optional). If None, auto-width.
        fig_height: Height of the combined figure (optional). If None, auto-height.
        show_shared_legend: Boolean, if True, attempts to show a shared legend
                             with minimal repetition of trace names.

    Returns:
        A Plotly Figure object containing all input figures as subplots with a
        more effectively managed legend.
    """

    n_rows = (len(list_of_figures) + n_cols - 1) // n_cols
    combined_fig = make_subplots(rows=n_rows, cols=n_cols,
                                  subplot_titles=[fig.layout.title.text for fig in list_of_figures])

    row_index = 1
    col_index = 1
    added_legend_names = set() # Keep track of legend names already added

    for figure in list_of_figures:
        for trace in figure.data:
            trace_name = trace.name # Get the trace name
            if show_shared_legend:
                if trace_name not in added_legend_names and trace_name is not None: # Check if name is new and not None
                    combined_fig.add_trace(trace, row=row_index, col=col_index)
                    added_legend_names.add(trace_name) # Add name to set
                else:
                    # If name already added or is None, add trace but hide legend item for this instance
                    trace.showlegend = False # Hide legend for this *specific* trace instance
                    combined_fig.add_trace(trace, row=row_index, col=col_index)
            else: # If not showing shared legend, just add all traces (individual subplot legends are already hidden)
                combined_fig.add_trace(trace, row=row_index, col=col_index)


        # **Preserve Original Axis Configurations:**
        if hasattr(figure.layout, 'xaxis'): # Check if original figure has xaxis layout
            combined_fig.update_xaxes(figure.layout.xaxis, row=row_index, col=col_index)
        if hasattr(figure.layout, 'yaxis'): # Check if original figure has yaxis layout
            combined_fig.update_yaxes(figure.layout.yaxis, row=row_index, col=col_index)

        # **Transfer Shapes (Hlines)**
        if figure.layout.shapes: # Check if shapes exist in the original figure
            for shape in figure.layout.shapes:
                combined_fig.add_shape(shape, row=row_index, col=col_index) # Add shape to subplot

        # Move to the next subplot position
        if col_index < n_cols:
            col_index += 1
        else:
            col_index = 1
            row_index += 1

    # Update layout for the combined figure
    layout_updates = {'title_text': title}
    if fig_width:
        layout_updates['width'] = fig_width
    if fig_height:
        layout_updates['height'] = fig_height
    if show_shared_legend:
        layout_updates['showlegend'] = True  # Enable overall legend for combined figure
    else:
        layout_updates['showlegend'] = False

    combined_fig.update_layout(**layout_updates)
    return combined_fig


def concat_data_lmer(folder_name, biomarker, ratio=30, pvalue_threshold=0.05):
    df_cross = pd.read_csv(f"./{folder_name}/statistic_estimation_crossover_{biomarker}_{ratio}.csv")
    df_cross = df_cross.drop("M", axis=1)
    df_cross[f"pvalue lower than {pvalue_threshold}"] = (df_cross["pvalue"] <= pvalue_threshold) & (df_cross["param"] < 0)
    # df_cross[f"pvalue lower than {pvalue_threshold}"] = (df_cross["pvalue"] <= pvalue_threshold)
    df_cross = df_cross.groupby(["sample_size"])[f"pvalue lower than {pvalue_threshold}"].aggregate(["sum", "count"]).reset_index()
    df_cross["Percentage"] = df_cross["sum"] / df_cross["count"]
    df_cross["Experiment type"] = "SEED design"
    
    df_paral = pd.read_csv(f"./{folder_name}/statistic_estimation_parallel_{biomarker}_{ratio}.csv")
    df_paral[f"pvalue lower than {pvalue_threshold}"] = (df_paral["pvalue"] <= pvalue_threshold) & (df_paral["param"] < 0)
    # df_paral[f"pvalue lower than {pvalue_threshold}"] = (df_paral["pvalue"] <= pvalue_threshold)
    df_paral = df_paral.groupby(["sample_size"])[f"pvalue lower than {pvalue_threshold}"].aggregate(["sum", "count"]).reset_index()
    df_paral["Percentage"] = df_paral["sum"] / df_paral["count"]
    df_paral["Experiment type"] = "Parallel design"
    df = pd.concat([df_cross, df_paral])
    df["Ratio"] = ratio / 100
    return df

if __name__ == "__main__":
    biomarker = "pTau217"
    # folder_name = "outputs_e0.25_m55"
    figs = []
    for efficacy in ["1.0", "0.50", "0.25"]:
        for m in ["55","33"]:
            folder_name = f"outputs_e{efficacy}_m{m}"
            dfs = [concat_data_lmer(biomarker=biomarker, folder_name=folder_name, ratio=i*10) for i in range(1,11)]
            df = pd.concat(dfs)
            df["Experiment type"].unique()
            fig = figure_plot(df, title=f"\u03B1={efficacy}, M={m[0]}")
            figs.append(fig)

    combined_figure_2col = combine_plotly_charts(
    figs, n_cols=2,
    title=f'{biomarker} Power Estimation',
    fig_width=1200, 
    fig_height=1000, 
    show_shared_legend=True)

    combined_figure_2col.write_image("./figs/pTau_combined_chart.pdf", scale=1)