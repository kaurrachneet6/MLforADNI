import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
import plotly.express as px
import plotly
import copy
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


from MachineLearningStreamlitBase.multiapp import MultiApp
from MachineLearningStreamlitBase.apps import streamlit_shapley_component

# add any app you like in apps directory
# from apps import topological_space
from apps import topological_space, select

app = MultiApp()
max_width = 4000
padding_top = 10
padding_right = 10
padding_left = 10
padding_bottom = 10
COLOR = 'black'
BACKGROUND_COLOR = 'white'
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

import copy

##TODO: UPDATE TITLE
st.title('Machine Learning for AD Subtype Prediction') 
app.add_app("Select", select.app)
app.add_app("SHAP Model Interpretation", streamlit_shapley_component.app)
app.add_app("Explore the ALS subtype topological space", topological_space.app)
app.run()