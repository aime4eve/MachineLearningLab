import numpy as np
import pandas as pd
import pickle
import logging
from collections import Counter,defaultdict

#one-hot encoding directly

# one-hot encoding directly
click = set()
hour = set(range(24))
C1 = set()
banner_pos = set()
site_category = set()
app_category = set()
device_type = set()
device_conn_type = set()
C15 = set()
C16 = set()
C18 = set()
C20 = set()


# one-encoding by frequency bucket
C14 = []
C17 = []
C19 = []
C21 = []
site_id = []
site_domain = []
app_id = []
app_domain = []
device_model = []
device_ip = []
device_id = []


direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type']

frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                              'device_model', 'device_id']

def initial_data(dataset,mode="test"):
    for data in dataset:
        click_v = set(data['click'].values)
        click = click | click_v

        C1_v = set(data['C1'].values)
        C1 = C1 | C1_v

        C15_v = set(data['C15'].values)
        C15 = C15 | C15_v

        C16_v = set(data['C16'].values)
        C16 = C16 | C16_v

        C18_v = set(data['C18'].values)
        C18 = C18 | C18_v

        C20_v = set(data['C20'].values)
        C20 = C20 | C20_v

        banner_pos_v = set(data['banner_pos'].values)
        banner_pos = banner_pos | banner_pos_v

        site_category_v = set(data['site_category'].values)
        site_category = site_category | site_category_v

        app_category_v = set(data['app_category'].values)
        app_category = app_category | app_category_v

        device_type_v = set(data['device_type'].values)
        device_type = device_type | device_type_v

        device_conn_type_v = set(data['device_conn_type'].values)
        device_conn_type = device_conn_type | device_conn_type_v

    with open('./../data/sets/click_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(click,f)

    with open('./../data/sets/hour_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(hour,f)

    with open('./../data/sets/C1_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(C1,f)

    with open('./../data/sets/C15_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(C15,f)

    with open('./../data/sets/C16_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(C16,f)

    with open('./../data/sets/C18_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(C18,f)

    with open('./../data/sets/C20_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(C20,f)

    with open('./../data/sets/banner_pos_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(banner_pos,f)

    with open('./../data/sets/site_category_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(site_category,f)

    with open('../data/sets/app_category_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(app_category,f)

    with open('./../data/sets/device_type_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(device_type,f)

    with open('./../data/sets/device_conn_type_{}.pkl'.format(mode),'wb') as f:
        pickle.dump(device_conn_type,f)


def load_fielddata_encoding(mode="test"):
    # loading frequency encoding fields
    # field2count dictionaries


    with open('./../data/field2count/C14_{}.pkl'.format(mode), 'rb') as f:
        C14 = pickle.load(f)

    with open('./../data/field2count/C17_{}.pkl'.format(mode), 'rb') as f:
        C17 = pickle.load(f)

    with open('./../data/field2count/C19_{}.pkl'.format(mode), 'rb') as f:
        C19 = pickle.load(f)

    with open('./../data/field2count/C21_{}.pkl'.format(mode), 'rb') as f:
        C21 = pickle.load(f)

    with open('./../data/field2count/site_id_{}.pkl'.format(mode), 'rb') as f:
        site_id = pickle.load(f)

    with open('./../data/field2count/site_domain_{}.pkl'.format(mode), 'rb') as f:
        site_domain = pickle.load(f)

    with open('./../data/field2count/app_id_{}.pkl'.format(mode), 'rb') as f:
        app_id = pickle.load(f)

    with open('./../data/field2count/app_domain_{}.pkl'.format(mode), 'rb') as f:
        app_domain = pickle.load(f)

    with open('./../data/field2count/device_model_{}.pkl'.format(mode), 'rb') as f:
        device_model = pickle.load(f)

    with open('./../data/field2count/device_id_{}.pkl'.format(mode), 'rb') as f:
        device_id = pickle.load(f)

    with open('./../data/sets/click_{}.pkl'.format(mode), 'rb') as f:
        click = pickle.load(f)

    with open('./../data/sets/hour_{}.pkl'.format(mode), 'rb') as f:
        hour = pickle.load(f)

    with open('./../data/sets/C1_{}.pkl'.format(mode), 'rb') as f:
        C1 = pickle.load(f)

    with open('./../data/sets/C15_{}.pkl'.format(mode), 'rb') as f:
        C15 = pickle.load(f)

    with open('./../data/sets/C16_{}.pkl'.format(mode), 'rb') as f:
        C16 = pickle.load(f)

    with open('./../data/sets/C18_{}.pkl'.format(mode), 'rb') as f:
        C18 = pickle.load(f)

    with open('./../data/sets/C20_{}.pkl'.format(mode), 'rb') as f:
        C20 = pickle.load(f)

    with open('./../data/sets/banner_pos_{}.pkl'.format(mode), 'rb') as f:
        banner_pos = pickle.load(f)

    with open('./../data/sets/site_category_{}.pkl'.format(mode), 'rb') as f:
        site_category = pickle.load(f)

    with open('./../data/sets/app_category_{}.pkl'.format(mode), 'rb') as f:
        app_category = pickle.load(f)

    with open('./../data/sets/device_type_{}.pkl'.format(mode), 'rb') as f:
        device_type = pickle.load(f)

    with open('./../data/sets/device_conn_type_{}.pkl'.format(mode), 'rb') as f:
        device_conn_type = pickle.load(f)

    field_dict = {}
    feature2field = {}
    field_index = 0
    ind = 0
    for field in direct_encoding_fields:
        # value to one-hot-encoding index dict
        fields_sets=eval(field)
        for value in list(fields_sets):
            field_dict[value]=ind
            feature2field[ind] = field_index
            ind+=1
        field_index+=1
    with open('./../data/dicts/' + field + '.pkl', 'wb') as f:
        pickle.dump(field_dict, f)

    for field in frequency_encoding_fields:
        # value to one-hot-encoding index dict
        field2count=eval(field)
        index_rare=None
        for k, count in field2count.items():
            if count<10:
                if index_rare==None:
                    field_dict[k]=ind
                    feature2field[ind]=field_index
                    index_rare=ind
                    ind+=1
                else:
                    field_dict[k]=index_rare
                    feature2field[index_rare]=field_index
            else:
                field_dict[k] = ind
                feature2field[ind] = field_index
                ind += 1
            field_index += 1
            with open('./../data/dicts/' + field + '.pkl', 'wb') as f:
                pickle.dump(field_dict, f)

    field_dict = {}
    field_sets = click
    for value in list(field_sets):
        field_dict[value] = ind + 1
        ind += 1

    with open('./../data/dicts/' + 'click' + '.pkl', 'wb') as f:
        pickle.dump(field_dict, f)

    with open('./../data/feature2field.pkl', 'wb') as f:
        pickle.dump(feature2field, f)


