import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta


try:
    st.sidebar.image("images/liquid_logo.png", width=250)
    st.sidebar.image("images/logo.png", width=250)
except Exception:
    pass

st.header("Arm Care available after next assesment to see deviation from baseline values (those taken 09/06/2025)")
