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

def app():
    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
    def load_model1():
        with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
            class_names = list(pickle.load(f))
        return class_names

    class_names = load_model1()

    st.write("## SHAP Model Interpretation")

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
    def load_model2():
        with open('saved_models/trainXGB_gpu.aucs', 'rb') as f:
            result_aucs = pickle.load(f)
        return  result_aucs

    result_aucs = load_model2()

    if len(result_aucs[class_names[0]]) == 3:
        df_res = pd.DataFrame({'class name': class_names, 'Train AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names], 'Test AUC (Replication)':  ["{:.2f}".format(result_aucs[i][1]) for i in class_names]})
        replication_avail = True
    else:
        df_res = pd.DataFrame({'class name': class_names, 'Train AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names], 'Test AUC':  ["{:.2f}".format(result_aucs[i][1]) for i in class_names]})
        replication_avail = False
    
    @st.cache(allow_output_mutation=True, ttl=24*3600)
    def get_shapley_value_data(train, replication=True, dict_map_result={}):
        dataset_type = '' 
        shap_values = np.concatenate([train[0]['shap_values_train'], train[0]['shap_values_test']], axis=0)
        X = pd.concat([train[1]['X_train'], train[1]['X_valid']], axis=0)
        exval = train[2]['explainer_train'] 
        auc_train = train[3]['AUC_train']
        auc_test = train[3]['AUC_test']
        ids = list(train[3]['ID_train'.format(dataset_type)]) + list(train[3]['ID_test'.format(dataset_type)])
        labels_pred = list(train[3]['y_pred_train'.format(dataset_type)]) + list(train[3]['y_pred_test'.format(dataset_type)]) 
        labels_actual = list(train[3]['y_train'.format(dataset_type)]) + list(train[3]['y_test'.format(dataset_type)]) 
        shap_values_updated = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
        train_samples = len(train[1]['X_train'])
        test_samples = len(train[1]['X_valid'])
        
        X.columns = ['({}) {}'.format(dict_map_result[col], col) if dict_map_result.get(col, None) is not None else col for col in list(X.columns)]
        shap_values_updated = copy.deepcopy(shap_values_updated) 
        patient_index = [hashlib.md5(str(s).encode()).hexdigest() for e, s in enumerate(ids)]
        return (X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_updated, train_samples, test_samples)
    
    st.write("## Introduction")
    st.write(
        """
        SHAP is an unified approach to explain the output of any supervised machine learning model. SHAP values are generated based on the idea that the change of an outcome to be explained with respect to a baseline can be attributed in different proportions to the model input features. In addition to assigning an importance value to every feature based on SHAP values, it shows the direction-of-effect at the level of the model as a whole. Furthermore, SHAP values provide both the global interpretability (i.e. collective SHAP values can show how much each predictor contributes) and local interpretability that explain why a sample receives its prediction. We built a surrogate XGBoost classification model to understand each individual genetic features’ effect on multiple the Neurodegenerative diseases classification. We randomly split the dataset into training (70%) and test (30%) sets. The model is trained on the training set and validated on the test set. The SHAP score is analyzed in detail to better understand the impact of features. Specifically, We used tree SHAP algorithm designed to provide human interpretable explanations for tree based learning models. 
        """
    )
    st.markdown(
        """<style>
        .boxBorder1 {
            outline-offset: 5px;
            font-size:20px;
        }</style>
        """, unsafe_allow_html=True) 
    # st.markdown('<div class="boxBorder1"><font color="black">Model Peformance</font></div>', unsafe_allow_html=True) 
    st.write("## Results")
    st.write("### Model Performance") 
    st.table(df_res.set_index('class name'))
    
    st.markdown('<div class="boxBorder1"><font color="black">Select the disease (positive class)</font></div>', unsafe_allow_html=True)
    feature_set_my = st.radio( "", class_names, index=0)

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
    def load_model3():
        with open('saved_models/trainXGB_gpu_{}.data'.format(feature_set_my), 'rb') as f:
            train = pickle.load(f)
        return train
    train = load_model3()
    st.write('---')
    st.markdown('<div class="boxBorder1"><font color="black">Showing Analysis for {}</font></div>'.format(feature_set_my), unsafe_allow_html=True) 
    st.write('---')
    data_load_state = st.text('Loading data...')
    # df = pd.read_csv('annotated_drugs.txt', sep='\t')
    # dict_map_result = dict(zip(list(df['MED_CODE']), list(df['meaning'])))
    dict_map_result = {}
    cloned_output = copy.deepcopy(get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result))
    data_load_state.text("Done Data Loading! (using st.cache)")
    X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output 
    
    
    import sklearn
    col0, col00 = st.beta_columns(2)
    with col0:
        st.write("### Data Statistics")
        st.info ('Total Features: {}'.format(X.shape[1]))
        st.info ('Total Samples: {} (Training: {}, Testing: {})'.format(X.shape[0], len_train, len_test))
    
    with col00:
        st.write("### ML Model Performance")
        st.info ('AUC Training Cohort: {}'.format(round(auc_train,2)))
        st.info ('AUC Testing Cohort: {}'.format( round(auc_test,2)))

    col01, col02 = st.beta_columns(2)
    with col01:
        st.write("### Training Cohort Confusion Matrix")
        Z = sklearn.metrics.confusion_matrix(labels_actual[:len_train], np.array(labels_pred[:len_train])>0.5)
        Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index= ['Actual 0', 'Actual 1'])
        st.table(Z_df)
    
    with col02:
        st.write("### Testing Cohort Confusion Matrix")
        Z = sklearn.metrics.confusion_matrix(labels_actual[len_train:], np.array(labels_pred[len_train:])>0.5)
        Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index= ['Actual 0', 'Actual 1'])
        st.table(Z_df)
    

    st.write('## Summary Plot')
    st.write("""Shows top-20 features that have the most significant impact on the classification model. In the figure, for some features lower value (blue color) corresponds to lower probability of the disease, when most of the blue colored points lie on the right side of baseline. On the other end, for some, lower values align with more healthy behaviour as blue colored points on the plot have negative impact on the model output. In this way, we can also observe that the directionality of different features.""")
    if st.checkbox("Show Summary Plot"):
        shap_type = 'trainXGB'
        col1, col2, col2111 = st.beta_columns(3)
        with col1:
            st.write('---')
            temp = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
            fig, ax = plt.subplots(figsize=(10,15))
            shap.plots.beeswarm(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns), show=False, max_display=20, order = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs, plot_size=0.47)# , return_objects=True 
            # shap.plots.beeswarm(temp, order=temp.mean(0).abs, show=False, max_display=20) # , return_objects=True 
            st.pyplot(fig)
            st.write('---')
        with col2:
            st.write('---')
            fig, ax = plt.subplots(figsize=(10,15))
            temp = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
            shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0), show=False, max_display=20, order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs)
            # shap.plots.bar(temp, order=temp.mean(0).abs, show=False, max_display=20)
            st.pyplot(fig)
            st.write('---')
        with col2111:
            st.write('---')
            fig, ax = plt.subplots(figsize=(10,15))
            temp = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
            shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).abs.mean(0), show=False, max_display=20, order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs)
            # shap.plots.bar(temp, order=temp.mean(0).abs, show=False, max_display=20)
            st.pyplot(fig)
            st.write('---')
    
    # st.write('## Dependence Plots')
    # st.write("""We can observe the interaction effects of different features in for predictions. To help reveal these interactions dependence_plot automatically lists (top-3) potential features for coloring.
    # Furthermore, we can observe the relationship betweem features and SHAP values for prediction using the dependence plots, which compares the actual feature value (x-axis) against the SHAP score (y-axis).
    # It shows that the effect of feature values is not a simple relationship where increase in the feature value leads to consistent changes in model output but a complicated non-linear relationship.""")
    # if st.checkbox("Show Dependence Plots"):
    #     feature_name = st.selectbox('Select a feature for dependence plot', options=list(X.columns))
    #     inds = shap.utils.potential_interactions(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)[:, feature_name], shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns))
    #
    #     st.write('Top3 Potential Interactions for ***{}***'.format(feature_name))
    #     col3, col4, col5 = st.beta_columns(3)
    #     with col3:
    #         shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[0]]))
    #         st.pyplot()
    #     with col4:
    #         shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[1]]))
    #         st.pyplot()
    #     with col5:
    #         shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[2]]))
    #         st.pyplot()

    labels_actual_new = np.array(labels_actual, dtype=np.float64)
    y_pred = (shap_values.sum(1) + exval) > 0
    misclassified = y_pred != labels_actual_new 
    st.write('## Decision Plots')
    st.write("""
        We selected 500 subsamples to understand the pathways of predictive modeling. SHAP decision plots show how complex models arrive at their predictions (i.e., how models make decisions). 
        Each observation’s prediction is represented by a colored line.
        At the top of the plot, each line strikes the x-axis at its corresponding observation’s predicted value. 
        This value determines the color of the line on a spectrum. 
        Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model’s base value. 
        This shows how each feature contributes to the overall prediction.
    """)
        # labels_pred_new = np.array(labels_pred, dtype=np.float)
        
    
    import random
    select_random_samples = np.random.choice(shap_values.shape[0], 500)

    new_X = X.iloc[select_random_samples]
    new_shap_values = shap_values[select_random_samples,:]
    new_labels_pred = np.array(labels_pred, dtype=np.float64)[select_random_samples] 


    st.write('### Pathways for Prediction (Hierarchical Clustering)')
    if st.checkbox("Show Prediction Pathways (order by: Feature Clustered)"):
        col3, col4, col5 = st.beta_columns(3)
        with col3:
            st.write('Typical Prediction Path: Uncertainity (0.4-0.6)')
            r = shap.decision_plot(exval, np.copy(new_shap_values), list(new_X.columns), feature_order='hclust', return_objects=True, show=False)
            T = new_X.iloc[(new_labels_pred >= 0.4) & (new_labels_pred <= 0.6)]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.copy(new_shap_values)[(new_labels_pred >= 0.4) & (new_labels_pred <= 0.6), :]
            
            fig, ax = plt.subplots()
            shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
            st.pyplot(fig)
        with col4:
            st.write('Typical Prediction Path: Positive Class (>=0.9)')
            fig, ax = plt.subplots()
            T = new_X.iloc[np.array(new_labels_pred, dtype=np.float64) >= 0.9]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.copy(new_shap_values)[new_labels_pred >= 0.9, :]
            shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
            st.pyplot(fig)
        with col5:
            st.write('Typical Prediction Path: Negative Class (<=0.1)')
            fig, ax = plt.subplots()
            T = new_X.iloc[new_labels_pred <= 0.1]
            import warnings
            with warnings.catch_warnings():
                   warnings.simplefilter("ignore")
                   sh = np.copy(new_shap_values)[new_labels_pred <= 0.1, :]
            shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
            st.pyplot(fig)
    

    st.write('### Pathways for Prediction (Feature Importance)')
    if st.checkbox("Show Prediction Pathways (order by: Feature Importance)"):
        col31, col41, col51 = st.beta_columns(3)
        with col31:
            st.write('Typical Prediction Path: Uncertainity (0.4-0.6)')
            r = shap.decision_plot(exval, np.copy(new_shap_values), list(new_X.columns), return_objects=True, show=False)
            T = new_X.iloc[(new_labels_pred >= 0.4) & (new_labels_pred <= 0.6)]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.copy(new_shap_values)[(new_labels_pred >= 0.4) & (new_labels_pred <= 0.6), :]
            fig, ax = plt.subplots()
            shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
            st.pyplot(fig)
        with col41:
            st.write('Typical Prediction Path: Positive Class (>=0.9)')
            fig, ax = plt.subplots()
            T = new_X.iloc[np.array(new_labels_pred, dtype=np.float64) >= 0.9]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.copy(new_shap_values)[new_labels_pred >= 0.9, :]
            shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
            st.pyplot(fig)
        with col51:
            st.write('Typical Prediction Path: Negative Class (<=0.1)')
            fig, ax = plt.subplots()
            T = new_X.iloc[new_labels_pred <= 0.1]
            import warnings
            with warnings.catch_warnings():
                   warnings.simplefilter("ignore")
                   sh = np.copy(new_shap_values)[new_labels_pred <= 0.1, :]
            shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
            st.pyplot(fig)
    
    st.write('### Pathways for Misclassified Samples')
    if st.checkbox("Show Misclassifies Pathways"):
        col6, col7 = st.beta_columns(2)
        with col6:
            st.info('Misclassifications (test): {}/{}'.format(misclassified[len_train:].sum(), len_test))
            fig, ax = plt.subplots()
            r = shap.decision_plot(exval, shap_values[misclassified], list(X.columns), link='logit', return_objects=True, new_base_value=0)
            st.pyplot(fig)
        with col7:
            # st.info('Single Example')
            sel_patients = [patient_index[e] for e, i in enumerate(misclassified) if i==1]
            select_pats = st.selectbox('Select misclassified patient id', options=list(sel_patients))
            id_sel_pats = sel_patients.index(select_pats)
            fig, ax = plt.subplots()
            shap.decision_plot(exval, shap_values[misclassified][id_sel_pats], X.iloc[misclassified,:].iloc[id_sel_pats], link='logit', feature_order=r.feature_idx, highlight=0, new_base_value=0)
            st.pyplot()