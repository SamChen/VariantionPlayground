import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.write("## Playground")

    with st.sidebar:
        with st.form("Options:"):
            sample_size = st.selectbox("Sample size: ", options=[18, 24, 36])
            st.form_submit_button("Submit")
            # apply_log = False

    df_stats = pd.read_csv(f"statistic_estimation_{sample_size}.csv")
    bar_chart = alt.Chart(df_stats).transform_joinaggregate(
        total='count(*)',
        groupby=["M"]
    ).transform_calculate(
        pct='1 / datum.total'
    ).mark_bar().encode(
        alt.X("pvalue:Q").bin(extent=[0, 1], step=0.05),
        alt.Y("sum(pct):Q", scale=alt.Scale(domain=[0, 1.0])).title("Percentage"),
        alt.Color("M:N")
    ).properties(
        width=200,
        height=200,
    ).facet(
        facet="M:N",
        columns=5
    ).resolve_scale(
        x='independent'
    ).properties(
        title=f"Sample size: {sample_size}",
    ).configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )
    st.altair_chart(bar_chart, theme="streamlit")
    # bar_chart.save(f"statistic_estimation_{sample_size}.png", ppi=400)
