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
from MachineLearningStreamlitBase.apps import streamlit_prediction_component, streamlit_shapley_component

# add any app you like in apps directory
# from apps import topological_space

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
st.title('Machine Learning for NDD using Drug Mining') 

app.add_app("Scientific background", streamlit_shapley_component.app)
app.add_app("Predict Patient ALS Subtype", streamlit_prediction_component.app)
##TODO: Add any apps you like
# app.add_app("Explore the ALS subtype topological space", topological_space.app)
app.run()

# import shap
# import joblib
# with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
#         class_names = list(pickle.load(f))
#     
# model = joblib.load( 'saved_models/trainXGB_gpu_{}.model'.format(class_names[0]) )
# import numpy as np
# d = {
#     'smoker': 0,
#     'cognitiveStatus2': np.nan,    
#     'elEscorialAtDx': 4,
#     'anatomicalLevel_at_onset': np.nan,
#     'site_of_onset': 0,
#     'onset_side': 0,
#     'ALSFRS1': 0,
#     'FVCPercentAtDx': 110,
#     'weightAtDx_kg': 57,
#     'rateOfDeclineBMI_per_month': 0,
#     'age_at_onset': 78.416667,
#     'firstALSFRS_daysIntoIllness': 195
# }
# d = {i:[j] for i,j in d.items()}
# X = pd.DataFrame(d, dtype=float)
# explainer_train = shap.TreeExplainer(model)
# shap_values_train = explainer_train.shap_values(X)
# st.write(shap_values_train)
# 
# shap.force_plot(explainer_train.expected_value, shap_values_train, X, show=False, matplotlib=True)
# st.pyplot()
# 
# fig, ax = plt.subplots()
# _ = shap.decision_plot(explainer_train.expected_value, shap_values_train, list(X.columns), link='logit', return_objects=True, new_base_value=0)
# st.pyplot(fig)
# 