import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.write("## Topological Space for AD Subtypes using NMF Approach")
    umap_org_full = pd.read_csv('saved_models/bl_m6_m12_features_m24NMF.csv', sep=',')
    colorable_columns_maps ={
        'adas__TOTSCORE___m12': 'ADAS MONTH 12 VISIT SCORE',
        'moca__moca_trail_making___bl': 'MOCA TRAIL MAKING BASELINE VISIT SCORE'
    }
    st.write(colorable_columns_maps)
    colorable_columns = list(colorable_columns_maps) 
    st.write(colorable_columns)
    st.write(set(colorable_columns).intersection(set(list(umap_org_full.columns))))
    colorable_columns = list(set(colorable_columns).intersection(set(list(umap_org_full.columns))))
    st.write("### Select a factor to color according to the factor")
    st.write(colorable_columns)
    st.write([colorable_columns_maps[i] for i in colorable_columns])
    select_color = st.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)
    umap_org_full = umap_org_full.rename(columns=colorable_columns_maps) 
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
    
