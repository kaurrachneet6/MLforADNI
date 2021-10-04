import pandas as pd
import pickle
import numpy as np
import os
from MachineLearningStreamlitBase.train_model import generateFullData

# Read Data File both pre-processed and raw data
# original_data = pd.read_csv("data/scriptToWrangleJessicaDataFreeze5/ALSregistry.AdrianoChio.wrangled.nodates.freeze5.csv")
# original_data = pd.read_csv('data/bl_features_m48_targets.csv')
Xdata = pd.read_csv('data/RFE_features_X.csv')
Ydata = pd.read_csv('data/Y.csv')
original_data = Ydata.merge(Xdata, left_on='RID', right_on='RID') 
selected_cols_x_top = [col for col in original_data.columns if not col in ['RID', 'predicted']] 
original_data = original_data.rename(columns={'RID': 'ID'})
dict_number = {
	0: 'ADvec1',
	1: 'ADvec2',
	2: 'ADvec3'
}
original_data['predicted'] = original_data['predicted'].map(lambda x: dict_number[x]) 
selected_cols_y = 'predicted'

Z_map = {
    'ADvec1':0,
    'ADvec2':1,
    'ADvec3':2,
}

obj = generateFullData()
original_encoded_data = original_data.copy() 
original_encoded_data[selected_cols_y] = original_encoded_data[selected_cols_y].map(lambda x: Z_map[x])
selected_raw_columns = [col for col in original_encoded_data.columns if not col in [selected_cols_y, 'ID'] ] 

os.makedirs('saved_models', exist_ok=True)
for class_name, ind in Z_map.items():
    print ('*'*30, class_name, '*'*30)
    original_encoded_data_temp = original_encoded_data.copy()
    original_encoded_data_temp[selected_cols_y] = original_encoded_data_temp[selected_cols_y].map (lambda x: 1 if x==ind else 0)
    # data_pol = pd.concat([data, data_rep], axis=0)
    model, train = obj.trainXGBModel_binaryclass(data=original_encoded_data_temp, feature_names=selected_raw_columns, label_name=selected_cols_y)
    with open('saved_models/RFE_trainXGB_gpu_{}.data'.format(class_name), 'wb') as f:
        pickle.dump(train, f)
    import joblib
    joblib.dump( model, 'saved_models/RFE_trainXGB_gpu_{}.model'.format(class_name) )
    # with open('saved_models/trainXGB_gpu_{}.model'.format(class_name), 'wb') as f:
    #     pickle.dump(model, f)

result_aucs = {}
for class_name in Z_map:
    with open('saved_models/RFE_trainXGB_gpu_{}.data'.format(class_name), 'rb') as f:
        temp = pickle.load(f)
    result_aucs[class_name] = (temp[3]['AUC_train'], temp[3]['AUC_test'] )
    print (class_name, result_aucs[class_name])

with open('saved_models/RFE_trainXGB_gpu.aucs', 'wb') as f:
    pickle.dump(result_aucs, f)

col_dict_map = {}
with open('saved_models/RFE_trainXGB_categorical_map.pkl', 'wb') as f:
    pickle.dump(col_dict_map, f)

with open('saved_models/RFE_trainXGB_class_map.pkl', 'wb') as f:
    pickle.dump(Z_map, f)
