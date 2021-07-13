import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.write("## Topological Space for AD Subtypes using NMF Approach")
    
    st.write("### Select the ADNI progression space")
    select_nmf = st.selectbox('', ['At 24th month after baseline', 'At 48th month after baseline'], index=0)
    
    if select_nmf == 'At 24th month after baseline':
        umap_org_full = pd.read_csv('saved_models/bl_m6_m12_features_m24NMF.csv', sep=',')
    else:
        umap_org_full = pd.read_csv('saved_models/bl_m6_m12_features_m48NMF.csv', sep=',')
    colorable_columns_maps ={
        'adas__TOTSCORE': 'ADAS TOTAL SCORE',
        'moca__moca_trail_making': 'MOCA TRAIL MAKING SCORE',
        'moca__moca_visuosoconstructional': 'MOCA VISUOSOCONSTRUCTIONAL SCORE',
        'moca__moca_naming': 'MOCA NAMING SCORE',
        'moca__moca_attention': 'MOCA ATTENTION SCORE',
        'moca__moca_immediate_recall': 'MOCA IMMEDIATE RECALL SCORE',
        'moca__moca_sen_repetetion': 'MOCA SENTENCE REPETETION SCORE',
        'moca__moca_fluency': 'MOCA FLUENCY SCORE',
        'moca__moca_abstraction': 'MOCA ABSTRACTION SCORE',
        'moca__moca_delayed_word_recall': 'MOCA DELAYED WORD RECALL SCORE',
        'moca__moca_orientation': 'MOCA ORIENTATION SCORE',
        'neurobat__CLOCKSCOR': 'NEUROBAT CLOCK SCORE',
        'neurobat__COPYSCOR': 'NEUROBAT COPY SCORE',
        'neurobat__LIMMTOTAL': 'NEUROBAT LIMM TOTAL SCORE',
        'npi_all__NPIATOT': 'NPIA TOTAL SCORE',
        'npi_all__NPIBTOT': 'NPIB TOTAL SCORE',
        'npi_all__NPICTOT': 'NPIC TOTAL SCORE',
        'npi_all__NPIDTOT': 'NPID TOTAL SCORE',
        'npi_all__NPIETOT': 'NPIE TOTAL SCORE',
        'npi_all__NPIFTOT': 'NPIF TOTAL SCORE',
        'npi_all__NPIGTOT': 'NPIG TOTAL SCORE',
        'npi_all__NPIHTOT': 'NPIH TOTAL SCORE',
        'npi_all__NPIITOT': 'NPII TOTAL SCORE',
        'npi_all__NPIJTOT': 'NPIJ TOTAL SCORE',
        'npi_all__NPIKTOT': 'NPIK TOTAL SCORE',
        'npi_all__NPILTOT': 'NPIL TOTAL SCORE',
        'mmse__MMSCORE': 'MMSE SCORE',
        'gd_scale__GDTOTAL': 'GD TOTAL SCORE',
        'ecogsp_memory': 'ECOGSP MEMORY SCORE',
        'ecogsp_lang': 'ECOGSP LANGUAGE SCORE',
        'ecogsp_vis': 'ECOGSP VISUAL SCORE',
        'ecogsp_plan': 'ECOGSP PLAN SCORE',
        'ecogsp_org': 'ECOGSP ORG SCORE',
        'ecogsp_division': 'ECOGSP DIVISION SCORE',
        'ecogpt_memory': 'ECOGPT MEMORY SCORE',
        'ecogpt_lang': 'ECOGPT LANGUAGE SCORE',
        'ecogpt_vis': 'ECOGPT VISUAL SCORE',
        'ecogpt_plan': 'ECOGPT PLAN SCORE',
        'ecogpt_org': 'ECOGPT ORG SCORE',
        'ecogpt_division': 'ECOGPT DIVISION SCORE',
        'cdr__CDMEMORY': 'CDR MEMORY SCORE',
        'cdr__CDORIENT': 'CDR ORIENT SCORE',
        'cdr__CDJUDGE': 'CDR JUDGE SCORE',
        'cdr__CDCOMMUN': 'CDR COMMUN SCORE',
        'cdr__CDHOME': 'CDR HOME SCORE',
        'cdr__CDCARE': 'CDR CARE SCORE',
        'FAQ__FAQTOTAL': 'FAQ TOTAL SCORE'     
    }
    
#     st.write(colorable_columns_maps)
    colorable_columns = list(colorable_columns_maps) 
#     st.write(colorable_columns)
#     st.write(set(colorable_columns).intersection(set(list(umap_org_full.columns))))
#     colorable_columns = list(set(colorable_columns).intersection(set(list(umap_org_full.columns))))
    st.write("### Select a feature to color according to the factor")
#     st.write(colorable_columns)
#     st.write([colorable_columns_maps[i] for i in colorable_columns])
    select_feature = st.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)
    
    visit_columns_maps ={
        'bl': 'BASELINE',
        'm06': 'MONTH 6',
        'm12': 'MONTH 12'
    }
#     st.write(colorable_columns_maps)
    visit_columns = list(visit_columns_maps) 
    
    st.write("### Select the corresponding visit")
#     st.write(colorable_columns)
#     st.write([colorable_columns_maps[i] for i in colorable_columns])
    select_visit = st.selectbox('', [visit_columns_maps[i] for i in visit_columns], index=0)
    
#     umap_org_full = umap_org_full.rename(columns=colorable_columns_maps) 
    feature_index = list(colorable_columns_maps.keys())[list(colorable_columns_maps.values()).index(select_feature)]
    visit_index = list(visit_columns_maps.keys())[list(visit_columns_maps.values()).index(select_visit)]
    select_color = feature_index + '___' + visit_index
    try:
        umap_org = umap_org_full[[select_color] + ['NMF_2_1', 'NMF_2_2']].dropna()
        color_discrete_map = {}
        color_discrete_map_list = ["red", "green", "blue", "magenta", "yellow", "pink", "grey", "black", "brown", "purple"]
        for e, classname in enumerate(sorted( list(set(umap_org[select_color]).union(set(umap_org[select_color]))) ) ) :
            color_discrete_map[classname] = color_discrete_map_list[e%10] 


        if len(color_discrete_map) < 10:
                # st.write('### Replication Cohort')
                fig = px.scatter(umap_org, x='NMF_2_1', y='NMF_2_2', color=select_color, color_discrete_map=color_discrete_map,  opacity=1, height=600, width=600)
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                    ))
                st.plotly_chart(fig, use_container_width=True)
        else:
                # st.write('### Replication Cohort')
                fig = px.scatter(umap_org, x='NMF_2_1', y='NMF_2_2', color=select_color,  opacity=1, height=600, width=600)
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                st.plotly_chart(fig, use_container_width=True)
    except:
        st.write("### This feature, visit pair does not exist in our database. Please try a different combination.")

