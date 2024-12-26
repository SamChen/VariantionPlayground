import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

from basic import *
eps = 1e-8

if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# SEED design experiments! 👋")

    st.write("## Estimate the power of SEED design through simulations")
    st.write('''
             ''')


