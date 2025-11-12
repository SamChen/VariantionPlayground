import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm

def figure_plot(df, title, plot_ci=False):
    """
    Generates a Plotly figure for power curves.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        title (str): The title for the subplot.
        plot_ci (bool): If True, plots the confidence interval as a shaded area.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    # Configuration for different experiment designs and sample sizes
    config_pairs = [
        ("SLIM design",     18, "#1f77b4"),
        ("Parallel design", 36, "#1f77b4"),
        ("SLIM design",     24, "#ff7f0e"),
        ("Parallel design", 48, "#ff7f0e"),
        ("SLIM design",     32, "#2ca02c"),
        ("Parallel design", 64, "#2ca02c"),
        ("SLIM design",     48, "#7f7f7f"),
        ("Parallel design", 96, "#7f7f7f"),
        ("Parallel design", 112, "#8c564b"),
        ("Parallel design", 130, "#e377c2"),
        ("Parallel design", 136, "#17becf"),
    ]

    for exp_type, sample_size, color in config_pairs:
        # Filter data for the current configuration
        data = df[(df["sample_size"] == sample_size) & (df["Experiment type"] == exp_type)].copy()
        if data.empty:
            continue

        # Ensure data is sorted by the x-axis value for correct plotting
        data.sort_values('Ratio', inplace=True)

        linestyle = "dash" if exp_type == "Parallel design" else None
        label = f"{exp_type} ({sample_size})"

        # Plot the confidence interval as a shaded area (plotted first to be in the background)
        if plot_ci and 'ci_lower' in data.columns and 'ci_upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=list(data['Ratio']) + list(data['Ratio'][::-1]), # X-axis forward, then backward
                y=list(data['ci_upper']) + list(data['ci_lower'][::-1]), # Upper bound, then lower bound reversed
                fill='toself',
                fillcolor=color,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'), # No border line for the fill
                hoverinfo="skip",
                showlegend=False,
                name=f"{label} CI"
            ))

        # Plot the main power curve line
        if linestyle:
            fig.add_trace(go.Scatter(
                x=data["Ratio"],
                y=data["Percentage"],
                name=label,
                line=dict(color=color, width=2, dash='dash')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data["Ratio"],
                y=data["Percentage"],
                name=label,
                line=dict(color=color, width=2)
            ))

    # Add a horizontal line at 80% power
    fig.add_hline(y=0.8, line=dict(color="red", width=1))

    # Update the layout of the figure
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(
            title='\u03B2', # Beta symbol
            tickmode='linear',
            tick0=0.0,
            dtick=0.1,
        ),
        yaxis=dict(
            title='Power',
            tickformat='.0%',
            tick0=0.0,
            dtick=0.1,
            range=[0, 1.05]
        ),
    )
    return fig

def combine_plotly_charts(list_of_figures, n_cols=2, title="Temp", fig_width=None, fig_height=None, show_shared_legend=False):
    """
    Combines a list of Plotly figures into a single figure with a grid layout,
    and attempts to create a shared legend (minimizing duplicate entries).

    Args:
        list_of_figures: A list of Plotly Figure objects.
        n_cols: The number of columns in the subplot grid (default is 2).
        title: The main title for the combined figure.
        fig_width: Width of the combined figure (optional).
        fig_height: Height of the combined figure (optional).
        show_shared_legend: If True, attempts to show a shared legend.

    Returns:
        A Plotly Figure object containing all input figures as subplots.
    """
    n_rows = (len(list_of_figures) + n_cols - 1) // n_cols
    combined_fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[fig.layout.title.text for fig in list_of_figures]
    )

    row_index = 1
    col_index = 1
    added_legend_names = set()

    for figure in list_of_figures:
        for trace in figure.data:
            trace_name = trace.name
            if show_shared_legend:
                if trace_name not in added_legend_names and trace_name is not None:
                    combined_fig.add_trace(trace, row=row_index, col=col_index)
                    added_legend_names.add(trace_name)
                else:
                    trace.showlegend = False
                    combined_fig.add_trace(trace, row=row_index, col=col_index)
            else:
                combined_fig.add_trace(trace, row=row_index, col=col_index)

        # Preserve original axis configurations
        if hasattr(figure.layout, 'xaxis'):
            combined_fig.update_xaxes(figure.layout.xaxis, row=row_index, col=col_index)
        if hasattr(figure.layout, 'yaxis'):
            combined_fig.update_yaxes(figure.layout.yaxis, row=row_index, col=col_index)

        # Transfer shapes (like hlines)
        if figure.layout.shapes:
            for shape in figure.layout.shapes:
                combined_fig.add_shape(shape, row=row_index, col=col_index)

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
    layout_updates['showlegend'] = show_shared_legend

    combined_fig.update_layout(**layout_updates)
    return combined_fig

def concat_data_lmer(folder_name, biomarker, ratio=30, pvalue_threshold=0.05, ci_level=0.95):
    """
    Reads and processes simulation data to calculate power and confidence intervals.

    Args:
        folder_name (str): Path to the folder containing the data.
        biomarker (str): Name of the biomarker.
        ratio (int): The ratio value for the dataset.
        pvalue_threshold (float): The threshold for statistical significance.
        ci_level (float): The confidence level for the interval (e.g., 0.95 for 95%).

    Returns:
        pd.DataFrame with power and confidence interval calculations.
    """
    # Process Crossover design data
    df_cross = pd.read_csv(f"./{folder_name}/statistic_estimation_crossover_{biomarker}_{ratio}.csv")
    df_cross = df_cross.drop("M", axis=1)
    df_cross[f"pvalue lower than {pvalue_threshold}"] = (df_cross["pvalue"] <= pvalue_threshold) & (df_cross["param"] < 0)
    df_cross = df_cross.groupby(["sample_size"])[f"pvalue lower than {pvalue_threshold}"].agg(["sum", "count"]).reset_index()
    df_cross["Percentage"] = df_cross["sum"] / df_cross["count"]
    df_cross["Experiment type"] = "SLIM design"

    # Process Parallel design data
    df_paral = pd.read_csv(f"./{folder_name}/statistic_estimation_parallel_{biomarker}_{ratio}.csv")
    df_paral[f"pvalue lower than {pvalue_threshold}"] = (df_paral["pvalue"] <= pvalue_threshold) & (df_paral["param"] < 0)
    df_paral = df_paral.groupby(["sample_size"])[f"pvalue lower than {pvalue_threshold}"].agg(["sum", "count"]).reset_index()
    df_paral["Percentage"] = df_paral["sum"] / df_paral["count"]
    df_paral["Experiment type"] = "Parallel design"

    # Combine dataframes
    df = pd.concat([df_cross, df_paral])
    df["Ratio"] = ratio / 100

    # --- Calculate Confidence Interval for the 'Percentage' (Power) ---
    # Using the Agresti-Coull interval, which is more accurate for proportions
    z = norm.ppf(1 - (1 - ci_level) / 2)  # Z-score for the desired confidence level
    n = df['count']
    p_hat = df['Percentage']

    # Adjusted estimates for the interval calculation
    n_adj = n + z**2
    p_adj = (n * p_hat + (z**2) / 2) / n_adj

    margin_of_error = z * np.sqrt(p_adj * (1 - p_adj) / n_adj)

    # Calculate lower and upper bounds
    df['ci_lower'] = p_adj - margin_of_error
    df['ci_upper'] = p_adj + margin_of_error

    return df

if __name__ == "__main__":
    biomarker = "pTau217"
    figs = []

    # Loop through different simulation scenarios
    for efficacy in ["1.0", "0.50", "0.25"]:
        for m in ["55", "33"]:
            folder_name = f"outputs_e{efficacy}_m{m}"

            # Concatenate data for different ratios
            dfs = [concat_data_lmer(biomarker=biomarker, folder_name=folder_name, ratio=i * 10) for i in range(1, 11)]
            df = pd.concat(dfs)

            # Create the plot for the current scenario with confidence intervals
            fig = figure_plot(df, title=f"\u03B1={efficacy}, M={m[0]}", plot_ci=True)
            figs.append(fig)

    # Combine all generated figures into a single subplot grid
    combined_figure_2col = combine_plotly_charts(
        figs,
        n_cols=2,
        title=f'{biomarker} Power Estimation',
        fig_width=1200,
        fig_height=1000,
        show_shared_legend=True
    )

    # Save the combined figure
    # You might need to install 'kaleido' for this to work: pip install kaleido
    try:
        combined_figure_2col.write_image("./figs/pTau_combined_chart.pdf", scale=1)
        print("Successfully saved figure to ./figs/pTau_combined_chart.pdf")
    except Exception as e:
        print(f"Could not save figure. You may need to install kaleido (`pip install kaleido`). Error: {e}")

    # To display the figure in an interactive environment (like a Jupyter notebook), you can just call:
    # combined_figure_2col.show()
