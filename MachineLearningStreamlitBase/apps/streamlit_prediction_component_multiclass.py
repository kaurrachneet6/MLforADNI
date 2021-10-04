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
import joblib
import xgboost as xgb

feature_mapping = {
    'neurobat__AVTOT5___m12': "Smoking status",
    'neurobat__AVTOT6___m06': "Cognitive status 2",    
    'neurobat__AVTOT6___m12': "El Escorial category at diagnosis",
    'neurobat__AVDEL30MIN___bl': "Anatomical level at onset",
    'neurobat__AVDEL30MIN___m06': "Site of symptom onset",
    'neurobat__AVDEL30MIN___m12': "Onset side",
    'neurobat__AVDELTOT___m06': "ALSFRS-R part 1 score",
    'neurobat__TRABSCOR___bl': "FVC% at diagnosis",
    'mmse__MMSCORE___m06': "Weight at diagnosis (kg)",
    'ecogsp___memory_bl': "Rate of BMI decline (per month)",
    'ecogsp___memory_m06': "Age at symptom onset",
    'ecogsp___memory_m12': "Time of first ALSFRS-R measurement (days from symptom onset)",
    'ecogsp___org_m12': "Age at symptom onset",
    'ecogsp___division_m12': "Age at symptom onset",
    'ecogpt___vis_m06': "Age at symptom onset",
    'FAQ__FAQTOTAL___m06': "Age at symptom onset",
    'FAQ__FAQTOTAL___m12': "Age at symptom onset",
    'adas__TOTAL13__bl': "Age at symptom onset",
    'adas__TOTAL13__m06': "Age at symptom onset",
    'adas__TOTAL13__m12': "Age at symptom onset"
}

def app():
    st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True) 
    st.markdown(
        """<style>
        .boxBorder {
            border: 2px solid #990066;
            padding: 10px;
            outline: #990066 solid 5px;
            outline-offset: 5px;
            font-size:25px;
        }</style>
        """, unsafe_allow_html=True) 
    st.markdown('<div class="boxBorder"><font color="RED">Disclaimer: This predictive tool is only for research purposes</font></div>', unsafe_allow_html=True)
    st.write("## Model Perturbation Analysis")
    with open('saved_models/RFE_trainXGB_class_map.pkl', 'rb') as f:
        class_names = list(pickle.load(f))
    
    M_dict = {}
    for classname in class_names:
        M_dict[classname] = joblib.load( 'saved_models/RFE_trainXGB_gpu_{}.model'.format(classname) )
    
    with open('saved_models/RFE_trainXGB_gpu_{}.data'.format(class_names[0]), 'rb') as f:
        train = pickle.load(f)
    with open('saved_models/RFE_trainXGB_categorical_map.pkl', 'rb') as f:
        col_dict_map = pickle.load(f)

    X = train[1]['X_valid'].copy() 
    ids = list(train[3]['ID_test'])
    X.index = ids
    labels_pred =  list(train[3]['y_pred_test']) 
    labels_actual = list(train[3]['y_test']) 
    # select_patient = st.selectbox("Select the patient", list(X.index), index=0)
    
    categorical_columns = []
    numerical_columns = []
    X_new = X.fillna('X')
    for col in X_new.columns:
        # if len(X_new[col].value_counts()) <= 10:
        if col_dict_map.get(col, None) is not None:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col) 
    
    st.write('### Please enter the following {} factors to perform prediction or select a random patient'.format(len(categorical_columns + numerical_columns)))
#     st.write("***Categorical Columns:***") 
#     st.write(categorical_columns)
#     st.write("***Numerical Columns:***") 
#     st.write(numerical_columns)
    from collections import defaultdict
    if st.button("Random Patient"):
        import random
        select_patient = random.choice(list(X.index))
    else:
        select_patient = list(X.index)[0]

    select_patient_index = ids.index(select_patient) 
    new_feature_input = defaultdict(list) 
    for key, val in col_dict_map.items():
        rval = {j:i for i,j in val.items()}
        X_new[key] = X_new[key].map(lambda x: rval.get(x, x))
    
    st.write('--'*10)
    st.write('##### Note: X denoted NA values')
    col1, col2, col3, col4 = st.beta_columns(4)
    for i in range(0, len(categorical_columns), 4):
        with col1:
            if (i+0) >= len(categorical_columns):
                continue
            c1 = categorical_columns[i+0] 
            idx = list(X_new[c1].unique()).index(X_new.loc[select_patient, c1]) 
            f1 = st.selectbox("{}".format(feature_mapping[c1]), list(X_new[c1].unique()), index=idx)
            new_feature_input[c1].append(col_dict_map[c1].get(f1, np.nan))
        with col2:
            if (i+1) >= len(categorical_columns):
                continue
            c2 = categorical_columns[i+1] 
            idx = list(X_new[c2].unique()).index(X_new.loc[select_patient, c2]) 
            f2 = st.selectbox("{}".format(feature_mapping[c2]), list(X_new[c2].unique()), index=idx)
            new_feature_input[c2].append(col_dict_map[c2].get(f2, np.nan))
        with col3:
            if (i+2) >= len(categorical_columns):
                continue
            c3 = categorical_columns[i+2] 
            idx = list(X_new[c3].unique()).index(X_new.loc[select_patient, c3]) 
            f3 = st.selectbox("{}".format(feature_mapping[c3]), list(X_new[c3].unique()), index=idx)
            new_feature_input[c3].append(col_dict_map[c3].get(f3, np.nan))
        with col4:
            if (i+3) >= len(categorical_columns):
                continue
            c4 = categorical_columns[i+3] 
            idx = list(X_new[c4].unique()).index(X_new.loc[select_patient, c4]) 
            f4 = st.selectbox("{}".format(feature_mapping[c4]), list(X_new[c4].unique()), index=idx)
            new_feature_input[c4].append(col_dict_map[c4].get(f4, np.nan))
    
    for col in numerical_columns:
#         st.write(X_new)
#         st.write(col)
        X_new[col] = X_new[col].map(lambda x: float(x) if not x=='X' else np.nan)
    for i in range(0, len(numerical_columns), 4):
        with col1:
            if (i+0) >= len(numerical_columns):
                continue
            c1 = numerical_columns[i+0] 
            idx = X_new.loc[select_patient, c1]
            f1 = st.number_input("{}".format(feature_mapping[c1]), min_value=X_new[c1].min(),  max_value=X_new[c1].max(), value=idx)
            new_feature_input[c1].append(f1)
        with col2:
            if (i+1) >= len(numerical_columns):
                continue
            c2 = numerical_columns[i+1] 
            idx = X_new.loc[select_patient, c2]
            f2 = st.number_input("{}".format(feature_mapping[c2]), min_value=X_new[c2].min(),  max_value=X_new[c2].max(), value=idx)
            new_feature_input[c2].append(f2)
        with col3:
            if (i+2) >= len(numerical_columns):
                continue
            c3 = numerical_columns[i+2] 
            idx = X_new.loc[select_patient, c3]
            f3 = st.number_input("{}".format(feature_mapping[c3]), min_value=X_new[c3].min(),  max_value=X_new[c3].max(), value=idx)
            new_feature_input[c3].append(f3)
        with col4:
            if (i+3) >= len(numerical_columns):
                continue
            c4 = numerical_columns[i+3] 
            idx = X_new.loc[select_patient, c4]
            f4 = st.number_input("{}".format(feature_mapping[c4]), min_value=X_new[c4].min(),  max_value=X_new[c4].max(), value=idx)
            new_feature_input[c4].append(f4)
    
    st.write('--'*10)
    st.write("### Do you want to see the effect of changing a factor on this patient?")
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    for e, classname in enumerate(class_names):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    show_whatif = st.checkbox("Enable what-if analysis")
    col01, col02 = st.beta_columns(2)
    with col01:
        st.write('### Prediction on actual feature values')
        feature_print = X_new.loc[select_patient, :].fillna('X')
        feature_print.index = feature_print.index.map(lambda x: feature_mapping[x])
        feature_print = feature_print.reset_index()
        feature_print.columns = ["Feature Name", "Feature Value"] 
        st.table(feature_print.set_index("Feature Name"))
        predicted_prob = defaultdict(list)
        predicted_class = -1
        max_val = -1
        for key, val in M_dict.items():
            predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(X.loc[select_patient, :].values.reshape(1, -1), feature_names=X.columns))[0])
            predicted_prob['classname'].append(key)
            if predicted_prob['predicted_probability'][-1] > max_val:
                predicted_class = key
                max_val = predicted_prob['predicted_probability'][-1] 
        K = pd.DataFrame(predicted_prob)
        K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
        K['color'] = ['zed' if i==predicted_class else 'red' for i in list(predicted_prob['classname']) ]
        # fig = px.bar(K, x='predicted_probability', y='classname', color='color', width=500, height=400, orientation='h')
        # # fig = px.bar(K, y='predicted_probability', x=sorted(list(predicted_prob['classname'])), width=500, height=400)
        # fig.update_layout(
        #     legend=None,
        #     yaxis_title="Class Labels",
        #     xaxis_title="Predicted Probability",
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=12,
        #         color="RebeccaPurple"
        #     ),
        #     margin=dict(l=10, r=10, t=10, b=10),
        # )
        # st.plotly_chart(fig)
        import altair as alt
        K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
        f = alt.Chart(K).mark_bar().encode(
                    y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=alt.X('Predicted Probability:Q'),
                    color=alt.Color('color', legend=None),
                ).properties(width=500, height=300)
        st.write(f)
        # st.write('#### Trajectory for Predicted Class')
        st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))
        with open('saved_models/RFE_trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
            new_train = pickle.load(f)
        exval = new_train[2]['explainer_train'] 
        explainer_train = shap.TreeExplainer(M_dict[predicted_class])
        t1 = pd.DataFrame(X.loc[select_patient, :]).T
        t2 = pd.DataFrame(X_new.loc[select_patient, :].fillna('X')).T
        shap_values_train = explainer_train.shap_values(t1)
        shap.force_plot(exval, shap_values_train, t1, show=False, matplotlib=True)
        st.pyplot()
        fig, ax = plt.subplots()
        r = shap.decision_plot(exval, shap_values_train, t2, link='logit', return_objects=True, new_base_value=0, highlight=0)
        st.pyplot(fig)
    if show_whatif:
        with col02:
            dfl = pd.DataFrame(new_feature_input)
            ndfl = dfl.copy()
            for key, val in col_dict_map.items():
                rval = {j:i for i,j in val.items()}
                ndfl[key] = ndfl[key].map(lambda x: rval.get(x, x))
            st.write('### Prediction with what-if analysis')

            feature_print_what = ndfl.iloc[0].fillna('X')
            feature_print_what.index = feature_print_what.index.map(lambda x: feature_mapping[x])
            feature_print_what = feature_print_what.reset_index()
            feature_print_what.columns = ["Feature Name", "Feature Value"] 
            selected = []
            for i in range(len(feature_print_what)):
                if feature_print.iloc[i]["Feature Value"] == feature_print_what.iloc[i]["Feature Value"]:
                    pass
                else:
                    selected.append(feature_print.iloc[i]["Feature Name"])

            # st.table(feature_print)
            st.table(feature_print_what.set_index("Feature Name").style.apply(lambda x: ['background: yellow' if (x.name in selected) else 'background: lightgreen' for i in x], axis=1))
            dfl = dfl[X.columns].replace('X', np.nan)
            predicted_prob = defaultdict(list)
            predicted_class = -1
            max_val = -1
            for key, val in M_dict.items():
                predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(dfl.iloc[0, :].values.reshape(1, -1), feature_names=dfl.columns))[0])
                predicted_prob['classname'].append(key)
                if predicted_prob['predicted_probability'][-1] > max_val:
                    predicted_class = key
                    max_val = predicted_prob['predicted_probability'][-1] 
            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i==predicted_class else 'red' for i in list(predicted_prob['classname']) ]
            import altair as alt
            K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
            f = alt.Chart(K).mark_bar().encode(
                y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=alt.X('Predicted Probability:Q'),
                    color=alt.Color('color', legend=None),
                ).properties( width=500, height=300)
            st.write(f)
            # fig = px.bar(K, x='predicted_probability', y='classname', color='color', width=500, height=400, orientation='h')
            # # fig = px.bar(K, y='predicted_probability', x=sorted(list(predicted_prob['classname'])), width=500, height=400)
            # fig.update_layout(
            # legend=None,
            # yaxis_title="Class Labels",
            # xaxis_title="Predicted Probability",
            # font=dict(
            #     family="Courier New, monospace",
            #     size=12,
            #     color="RebeccaPurple"
            # ),
            # margin=dict(l=10, r=10, t=10, b=10),
            # )  
            # st.plotly_chart(fig)
            st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))
            with open('saved_models/RFE_trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
                new_train = pickle.load(f)
            exval = new_train[2]['explainer_train'] 
            explainer_train = shap.TreeExplainer(M_dict[predicted_class])
            t1 = dfl.copy() 
            shap_values_train = explainer_train.shap_values(t1)
            shap.force_plot(exval, shap_values_train, t1, show=False, matplotlib=True)
            st.pyplot()
            fig, ax = plt.subplots()
            _ = shap.decision_plot(exval, shap_values_train, ndfl.fillna('X'), link='logit', feature_order=r.feature_idx, return_objects=True, new_base_value=0, highlight=0)
            st.pyplot(fig)
    
    # st.write('### Force Plots')
    # patient_name = st.selectbox('Select patient id', options=list(patient_index))
    # sample_id = patient_index.index(patient_name)
    # col8, col9 = st.beta_columns(2)
    # with col8:
    #     st.info('Actual Label: ***{}***'.format('PD' if labels_actual[sample_id]==1 else 'HC'))
    #     st.info('Predicted PD class Probability: ***{}***'.format(round(float(labels_pred[sample_id]), 2)))
    # with col9:
    #     shap.force_plot(exval, shap_values[sample_id,:], X.iloc[sample_id,:], show=False, matplotlib=True)
    #     st.pyplot()
    
    # col10, col11 = st.beta_columns(2)
    # with col10:
    #     fig, ax = plt.subplots()
    #     shap.decision_plot(exval, shap_values[sample_id], X.iloc[sample_id], link='logit', highlight=0, new_base_value=0)
    #     st.pyplot()



# fig = px.pie(pd.DataFrame(predicted_prob), values='predicted_probability', names='classname', color='classname', color_discrete_map=color_discrete_map)
        # fig.update_layout(legend=dict(
        #         yanchor="top",
        #         y=0.99,
        #         xanchor="right",
        #         x=1.05
        #     ))
