# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:07:45 2018

@author: STP
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import xgboost as xgb

def handle_dict(dic, col1, col2=None, cell=2):
    data = pd.DataFrame(np.zeros((0,0),dtype=float))
    for i in dic.keys():
        temp = []
        if cell == 1:
            temp.append([i, dic[i]])
        else:
            for j in dic[i].keys():
                temp.append([i,j,dic[i][j]])
        temp = pd.DataFrame(temp)
        data = pd.concat([data, temp], axis=0)
    if cell == 1:
        data.columns = [col1, col1+'_data']
    else:
        data.columns = [col1, col2, col1+'_data']
    return data
        
def rename_data(df, i, j, col):
    globals()[col+str(i)+'_'+str(j)] = df
    return col+str(i)+'_'+str(j)

def get_save_id(dictionary, limit):
    dictionary_drop = pd.DataFrame(list(dictionary.items()), columns=['id', 'num'])
    item_save_id = list(dictionary_drop[dictionary_drop['num']>limit]['id'])
    return item_save_id

def get_merge_data(dictionary, save_id, col_name):
    middle = []
    for i in dictionary.keys():
        new = {}
        for j in dictionary[i].keys():
            if j in save_id:
                name = col_name + str(j)
                new[name] = dictionary[i][j]
        middle.append(new)
    merge = pd.DataFrame(middle)
    return merge

def get_merge_data_new(data, dictionary, save_id, col_name, which_type):
    middle = []
    for index, row in data.iterrows():
        new = {}
        new['index'] = index
        if which_type == 'p_c':
            if row['predict_category_property'] == '-1':
                middle.append(new)
            else:
                each = [x.split(':')[1] for x in row['predict_category_property'].split(';')]
                for i in each:
                    each_k = [x for x in i.split(',')]
                    for j in each_k:
                        if j in save_id:
                            name = col_name + str(j)
                            new[name] = dictionary[i]
                middle.append(new)
        else:
            if which_type == 'i_p':
                each = [x for x in row['item_property_list'].split(';')]
            if which_type == 'p_c_single':
                each = [x.split(':')[0] for x in row['predict_category_property'].split(';')]
            for i in each:
                if i in save_id:
                    name = col_name + str(i)
                    new[name] = dictionary[i]
            middle.append(new)
    merge = pd.DataFrame(middle)
    return merge


path = 'C:\\Users\\STP\\Desktop\\code\\'
output_path = '/root/ijcai-18/'
go_pred = 1

train = pd.read_csv(path+'round1_ijcai_18_train_20180301.txt', engine='python', sep=' ')
test = pd.read_csv(path+'round1_ijcai_18_test_a_20180301.txt', engine='python', sep=' ')

train = train.drop_duplicates()

data = pd.concat([train, test], axis=0).reset_index(drop=True)

print ('data merge completed')

#brand
brand = pd.DataFrame(data['item_brand_id'])
brand['sum'] = 1
brand = brand.groupby('item_brand_id').sum().reset_index()
brand_save_id = list(brand[brand['sum'] > 100]['item_brand_id'])
data['item_brand_id'] = data['item_brand_id'].apply(lambda x: x if x in brand_save_id else np.nan)

del brand, brand_save_id

#city
city = pd.DataFrame(data['item_city_id'])
city['sum'] = 1
city = city.groupby('item_city_id').sum().reset_index()
city_save_id = list(city[city['sum'] > 100]['item_city_id'])
data['item_city_id'] = data['item_city_id'].apply(lambda x: x if x in city_save_id else np.nan)

del city, city_save_id

#user
user_id = data['user_id'].drop_duplicates()
print ('total user number:', len(user_id))
user_sum = []
for n, i in enumerate(user_id):
    data_temp = data[data['user_id'] == i][['instance_id', 'context_timestamp', 'is_trade']].sort_values('context_timestamp')
    user_sum.append([data_temp['instance_id'].iloc[0], 0, 0, 0, 0, 0])
    if len(data_temp) > 1:
        for j in range(len(data_temp)-1):
            data_temp2 = data_temp.iloc[:(j+2),:].copy()
            data_temp2['context_timestamp'] = data_temp2['context_timestamp'].iloc[-1] - data_temp2['context_timestamp']
            one_minute = sum(data_temp2['context_timestamp']<=60)
            ten_minute = sum(data_temp2['context_timestamp']<=600)
            one_hour = sum(data_temp2['context_timestamp']<=3600)
            one_day = sum(data_temp2['context_timestamp']<=86400)
            two_day = sum(data_temp2['context_timestamp']<=172800)
            user_sum.append([data_temp2['instance_id'].iloc[-1], one_minute, ten_minute, one_hour, one_day, two_day])
    if n % 10000 == 0:
        print (n, 'completed')
user_sum = pd.DataFrame(user_sum)
user_sum.columns = ['instance_id', 'one_minute_sum', 'ten_minute_sum', 'one_hour_sum', 'one_day_sum', 'two_day_sum']
data = pd.merge(data, user_sum, on='instance_id', how='left')
del user_sum

#context_timestamp
data['context_timestamp_day'] = data.context_timestamp.apply(lambda x:time.localtime(x).tm_mday)
data['context_timestamp_hour'] = data.context_timestamp.apply(lambda x:time.localtime(x).tm_hour)
del data['context_timestamp']

print ('context_timestamp handle completed')
        
#item_category_list
data['item_category_list_1'] = data.item_category_list.apply(lambda x:x.split(';')[1])
merge_category = pd.get_dummies(data['item_category_list_1'], prefix='item_category')
del data['item_category_list']

print ('item_category_list handle completed')

#1,2,3day handle
day_min = data['context_timestamp_day'].min()
day_max = data['context_timestamp_day'].max()
time_rate_1 = pd.DataFrame(np.zeros((0,0),dtype=float))
time_rate_2 = pd.DataFrame(np.zeros((0,0),dtype=float))
item_category_rate_1 = pd.DataFrame(np.zeros((0,0),dtype=float))
item_category_rate_2 = pd.DataFrame(np.zeros((0,0),dtype=float))
for i in [1,2]:
    for j in range(day_min+i, day_max+1):
        time_rate = pd.DataFrame(np.zeros((0,0),dtype=float))
        item_category_rate = pd.DataFrame(np.zeros((0,0),dtype=float))
        item_property_temp = defaultdict(lambda : 0)
        item_property_rate_temp = defaultdict(lambda : 0)
        predict_category_temp = defaultdict(lambda : 0)
        predict_category_temp_rate = defaultdict(lambda : 0)
        predict_category_temp_single = defaultdict(lambda : 0)
        predict_category_temp_single_rate = defaultdict(lambda : 0)
        item_property_list_each = defaultdict(lambda : defaultdict(lambda : 0))
        item_property_dict = defaultdict(lambda : 0)
        predict_category_property_each = defaultdict(lambda : defaultdict(lambda : 0))
        predict_category_property_dict = defaultdict(lambda : 0)
        
        data_temp = data[(data['context_timestamp_day'] < j) & (data['context_timestamp_day'] >= (j-i))]
        #time_rate
        time_rate_temp = data_temp[['context_timestamp_hour', 'is_trade']].groupby('context_timestamp_hour').sum() / data_temp[['context_timestamp_hour', 'is_trade']].groupby('context_timestamp_hour').count()
        time_rate_temp.columns = [str(i)+'_day_time_rate']
        time_rate_temp = time_rate_temp.reset_index()
        time_rate_temp['context_timestamp_day'] = j
        time_rate = pd.concat([time_rate, time_rate_temp], axis=0).reset_index(drop=True)
        if i == 1:
            time_rate_1 = pd.concat([time_rate_1, time_rate], axis=0).reset_index(drop=True)
        elif i == 2:
            time_rate_2 = pd.concat([time_rate_2, time_rate], axis=0).reset_index(drop=True)
        #item_category_list_rate
        item_category_temp = data_temp[['item_category_list_1', 'is_trade']].groupby('item_category_list_1').sum() / data_temp[['item_category_list_1', 'is_trade']].groupby('item_category_list_1').count()
        item_category_temp.columns = [str(i)+'_day_item_category_rate']
        item_category_temp = item_category_temp.reset_index()
        item_category_temp['context_timestamp_day'] = j
        item_category_rate = pd.concat([item_category_rate, item_category_temp], axis=0).reset_index(drop=True)
        if i == 1:
            item_category_rate_1 = pd.concat([item_category_rate_1, item_category_rate], axis=0).reset_index(drop=True)
        elif i == 2:
            item_category_rate_2 = pd.concat([item_category_rate_2, item_category_rate], axis=0).reset_index(drop=True)
        #item_property_list_rate
        for index, row in data_temp.iterrows():
            each_i = [x for x in row['item_property_list'].split(';')]
            for k in each_i:
                item_property_temp[k] += 1
                if row['is_trade'] == 1:
                    item_property_rate_temp[k] += 1
#        for k in item_property_rate_temp.keys():
#            item_property_rate_temp[k] /= item_property_temp[k]
#       item_property_temp = pd.DataFrame(list(item_property_temp.items()), columns=['item_property_id', 'item_property_num'])
#       item_property_rate_temp = pd.DataFrame(list(item_property_rate_temp.items()), columns=['item_property_id', 'item_property_rate'])
        #predict_category_property_rate
        for index, row in data_temp.iterrows():
            if row['predict_category_property'] != '-1':
                each_i = [x.split(':')[0] for x in row['predict_category_property'].split(';')]
                each_j = [x.split(':')[1] for x in row['predict_category_property'].split(';')]
                for k, l in zip(each_i, each_j):
                    predict_category_temp_single[k] += 1
                    if row['is_trade'] == 1:
                        predict_category_temp_single_rate[k] += 1
                    each_k = [x for x in l.split(',')]
                    for m in each_k:
                        predict_category_temp[m] += 1
                        if row['is_trade'] == 1:
                            predict_category_temp_rate[m] += 1

        for index, row in data_temp.iterrows():
            each_i = [x for x in row['item_property_list'].split(';')]
#            each_j = [x.split(':')[0] for x in row['predict_category_property'].split(',')]
            for m in each_i:
                item_property_dict[m] += 1
                item_property_list_each[index][m] = 1
#            for m in each_j:
#                predict_category_property_dict[m] += 1
#                predict_category_property_each[index][m] = 1
#        item_property_drop = pd.DataFrame(list(item_property_dict.items()), columns=['item_property_id', 'item_property_num'])
#        item_save_id = list(item_property_drop[item_property_drop['item_property_num']>100]['item_property_id'])
#        globals()['i_p_'+str(i)+'_'+str(j)] = item_save_id
#        predict_category_property = pd.DataFrame(list(predict_category_property_dict.items()), 
#                                                 columns=['predict_category_property_id', 'predict_category_property_num'])
#        predict_category_save_id = list(predict_category_property[predict_category_property['predict_category_property_num']>2000]['predict_category_property_id'])
        globals()['i_p_'+str(i)+'_'+str(j)] = item_property_dict
        globals()['i_p_each_'+str(i)+'_'+str(j)] = item_property_list_each
#        globals()['p_c_'+str(i)+'_'+str(j)] = predict_category_property_dict
#        globals()['p_c_each_'+str(i)+'_'+str(j)] = predict_category_property_each

        globals()['p_c_'+str(i)+'_'+str(j)] = predict_category_temp
        globals()['p_c_each_'+str(i)+'_'+str(j)] = predict_category_temp_rate
        globals()['p_c_single_'+str(i)+'_'+str(j)] = predict_category_temp_single
        globals()['p_c_single_each_'+str(i)+'_'+str(j)] = predict_category_temp_single_rate

#        predict_category_temp = handle_dict(predict_category_temp, 'predict_category_leimu', 'predict_category_shuxing', 2)
#        predict_category_temp_rate = handle_dict(predict_category_temp_rate, 'predict_category_rate_leimu', 'predict_category_rate_shuxing', 2)
#        predict_category_temp_single = handle_dict(predict_category_temp_single, 'predict_category_leimu', cell=1)
#        predict_category_temp_single_rate = handle_dict(predict_category_temp_single_rate, 'predict_category_rate_leimu', cell=1)

#        time_rate.to_csv(path+str(i)+'_'+str(j)+'day_time_rate.csv', index=None)
#        item_category_rate.to_csv(path+str(i)+'_'+str(j)+'day_item_category_rate.csv', index=None)
        
#        rename_data(item_property_temp, i, j, 'item_property_num')
#        rename_data(item_property_rate_temp, i, j, 'item_property_rate')
#        rename_data(predict_category_temp, i, j, 'predict_category')
#        rename_data(predict_category_temp_rate, i, j, 'predict_category_rate')
#        rename_data(predict_category_temp_single, i, j, 'predict_category_single')
#        rename_data(predict_category_temp_single_rate, i, j, 'predict_category_single_rate')
        
#        item_property_temp.to_csv(path+str(i)+'_'+str(j)+'day_item_property_num.csv', index=None)
#        item_property_rate_temp.to_csv(path+str(i)+'_'+str(j)+'day_item_property_rate.csv', index=None)
#        predict_category_temp.to_csv(path+str(i)+'_'+str(j)+'day_predict_category.csv', index=None)
#        predict_category_temp_rate.to_csv(path+str(i)+'_'+str(j)+'day_predict_category_rate.csv', index=None)
#        predict_category_temp_single.to_csv(path+str(i)+'_'+str(j)+'day_predict_category_single.csv', index=None)
#        predict_category_temp_single_rate.to_csv(path+str(i)+'_'+str(j)+'day_predict_category_single_rate.csv', index=None)
        
        print (i, j, 'completed')
        
data = pd.merge(data, time_rate_1, on=['context_timestamp_hour', 'context_timestamp_day'], how='left')
data = pd.merge(data, item_category_rate_1, on=['item_category_list_1', 'context_timestamp_day'], how='left')
data = pd.merge(data, time_rate_2, on=['context_timestamp_hour', 'context_timestamp_day'], how='left')
data = pd.merge(data, item_category_rate_2, on=['item_category_list_1', 'context_timestamp_day'], how='left')

del data['item_category_list_1'], time_rate, item_category_rate, item_property_temp, time_rate_temp, item_category_temp

print ('time rate merge completed and generated data for merge')
'''
#convert into dict
handle_list = ['item_property_num','item_property_rate', 'predict_category',
               'predict_category_rate', 'predict_category_single', 'predict_category_single_rate']
for i in [1,2]:
    for j in range(day_min+i, day_max+1):
        for k in handle_list:
            data_temp = eval(str(k)+str(i)+'_'+str(j))
            if k == 'predict_category' or k == 'predict_category_rate':
                init_dict = defaultdict(lambda : defaultdict(lambda : 0))
                col1 = data_temp.columns[0]
                col2 = data_temp.columns[1]
                for index, row in data_temp.iterrows():
                    init_dict[int(row.iloc[0])][int(row.iloc[1])] =float(row.iloc[2])
                globals()[str(k)+str(i)+'_'+str(j)] = init_dict
            else:
                init_dict = defaultdict(lambda : 0)
                col1 = data_temp.columns[0]
                for index, row in data_temp.iterrows():
                    init_dict[int(row.iloc[0])] =float(row.iloc[1])
                globals()[str(k)+str(i)+'_'+str(j)] = init_dict
'''
'''
#item_property_list
item_property_list_each = defaultdict(lambda : defaultdict(lambda : 0))
item_property_dict = defaultdict(lambda : 0)
for index, row in data.iterrows():
    each_i = [x for x in row['item_property_list'].split(';')]
    for j in each_i:
        item_property_dict[j] += 1
        item_property_list_each[index][j] = 1
item_property = pd.DataFrame(list(item_property_dict.items()), columns=['item_property_id', 'item_property_num'])
item_save_id = list(item_property[item_property['item_property_num']>10000]['item_property_id'])
del item_property
middle = []
for i in item_property_list_each.keys():
    new = {}
    for j in item_property_list_each[i].keys():
        if j in item_save_id:
            for m in [1,2]:
                for n in range(day_min+2, day_max+1):
                    name = 'i_p_' + str(m)+'_'+str(n)+'_'+str(j)
                    new[name] = item_property_list_each[i][j]
    middle.append(new)
merge1 = pd.DataFrame(middle)
data = pd.concat([data, merge1], axis=1)
del middle, item_property_list_each, merge1, item_save_id, item_property_dict

print ('item_property_list handle completed')

#predict_category_property
predict_category_property_list = []
middle_dict = defaultdict(lambda : defaultdict(lambda : 0))
predict_category_property_dict = defaultdict(lambda : 0)
for index, row in data.iterrows():
    each_i = [x.split(':')[0] for x in row['predict_category_property'].split(',')]
    for i in each_i:
        predict_category_property_dict[i] += 1
        middle_dict[index][i] = 1
predict_category_property = pd.DataFrame(list(predict_category_property_dict.items()), 
                                         columns=['predict_category_property_id', 'predict_category_property_num'])
predict_category_save_id = list(predict_category_property[predict_category_property['predict_category_property_num']>2000]['predict_category_property_id'])
del predict_category_property, predict_category_property_list
middle = []
for i in middle_dict.keys():
    new = {}
    for j in middle_dict[i].keys():
        if j in predict_category_save_id:
            name = 'predict_category_property' + str(j)
            new[name] = middle_dict[i][j]
    middle.append(new)
merge1 = pd.DataFrame(middle)
data = pd.concat([data, merge1], axis=1)
del middle, middle_dict, merge1, predict_category_save_id, predict_category_property_dict
i_p_each_1_19[9148482949976129397]
print ('predict_category_property handle completed')
'''

#
i_p_1_set = set(i_p_1_20)
#i_p_each_1_set = set()
p_c_1_set = set(p_c_1_20)
p_c_single_1_set = set(p_c_single_1_20)
j = 1
for i in range(day_min+2, day_max+1):
#    data_temp = data[(data['context_timestamp_day'] < (i+3)) & (data['context_timestamp_day'] >= i)
    for k in ['i_p_','p_c_','p_c_single_']:
        if k == 'i_p_':
            i_p_1_set = list(set(i_p_1_set) & set(get_save_id(globals()['i_p_'+str(j)+'_'+str(i)], 1000)))
#        if k == 'i_p_each_':
#            i_p_each_1_set = list(set(i_p_each_1_set) | set(get_save_id(globals()['i_p_each_'+str(j)+'_'+str(i)], 1000)))
        if k == 'p_c_':
            p_c_1_set = list(set(p_c_1_set) & set(get_save_id(globals()['p_c_'+str(j)+'_'+str(i)], 1000)))
        if k == 'p_c_single_':
            p_c_single_1_set = list(set(p_c_single_1_set) & set(get_save_id(globals()['p_c_single_'+str(j)+'_'+str(i)], 1000)))

i_p_2_set = set(i_p_2_20)
#i_p_each_2_set = set()
p_c_2_set = set(p_c_2_20)
p_c_single_2_set = set(p_c_single_2_20)
j = 2
for i in range(day_min+2, day_max+1):
#    data_temp = data[(data['context_timestamp_day'] < (i+3)) & (data['context_timestamp_day'] >= i)
    for k in ['i_p_','p_c_','p_c_single_']:
        if k == 'i_p_':
            i_p_2_set = list(set(i_p_2_set) & set(get_save_id(globals()['i_p_'+str(j)+'_'+str(i)], 1000)))
#        if k == 'i_p_each_':
#            i_p_each_2_set = list(set(i_p_each_2_set) | set(get_save_id(globals()['i_p_each_'+str(j)+'_'+str(i)], 1000)))
        if k == 'p_c_':
            p_c_2_set = list(set(p_c_2_set) & set(get_save_id(globals()['p_c_'+str(j)+'_'+str(i)], 1000)))
        if k == 'p_c_single_':
            p_c_single_2_set = list(set(p_c_single_2_set) & set(get_save_id(globals()['p_c_single_'+str(j)+'_'+str(i)], 1000)))

data_new = pd.DataFrame(np.zeros((0,0),dtype=float))
for i in range(day_min+2, day_max+1):
    data_temp = data[data['context_timestamp_day'] == i]
    for j in [1, 2]:
        for k in ['i_p_','i_p_each_','p_c_','p_c_each_','p_c_single_','p_c_single_each_']:
            if k == 'i_p_':
                merge = get_merge_data_new(data_temp, globals()['i_p_'+str(j)+'_'+str(i)], globals()['i_p_'+str(j)+'_set'], str(j)+'_i_p_', 'i_p').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
            if k == 'i_p_each_':
                merge = get_merge_data_new(data_temp, globals()['i_p_each_'+str(j)+'_'+str(i)], globals()['i_p_'+str(j)+'_set'], str(j)+'_i_p_each_', 'i_p').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
            if k == 'p_c_':
                merge = get_merge_data_new(data_temp, globals()['p_c_'+str(j)+'_'+str(i)], globals()['p_c_'+str(j)+'_set'], str(j)+'_p_c_', 'p_c').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
            if k == 'p_c_each_':
                merge = get_merge_data_new(data_temp, globals()['p_c_each_'+str(j)+'_'+str(i)], globals()['p_c_'+str(j)+'_set'], str(j)+'_p_c_each_', 'p_c').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
            if k == 'p_c_single_':
                merge = get_merge_data_new(data_temp, globals()['p_c_single_'+str(j)+'_'+str(i)], globals()['p_c_single_'+str(j)+'_set'], str(j)+'_p_c_single_', 'p_c_single').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
            if k == 'p_c_single_each_':
                merge = get_merge_data_new(data_temp, globals()['p_c_single_each_'+str(j)+'_'+str(i)], globals()['p_c_single_'+str(j)+'_set'], str(j)+'_p_c_single_each_', 'p_c_single').set_index('index')
                data_temp = pd.concat([data_temp, merge], axis=1)
    data_new = pd.concat([data_new, data_temp], axis=0)
#    len(data_temp.columns)
#    len(data_temp.columns.drop_duplicates())
#    for m in list(data_new.columns):
#        if m not in list(data_new.columns.drop_duplicates(keep=False)):
#            print (m)
#    data_new.columns
    print ('day:', i, 'completed')

data = data_new.copy()
del data_new
#handle
merge1 = pd.get_dummies(data['item_city_id'], prefix='item_city')
merge2 = pd.get_dummies(data['item_brand_id'], prefix='item_brand')
merge3 = pd.get_dummies(data['context_page_id'], prefix='context_page')
merge4 = pd.get_dummies(data['item_collected_level'], prefix='item_collected')
merge5 = pd.get_dummies(data['item_pv_level'], prefix='item_pv')
merge6 = pd.get_dummies(data['item_price_level'], prefix='item_price')
merge7 = pd.get_dummies(data['item_sales_level'], prefix='item_sales')
merge8 = pd.get_dummies(data['shop_review_num_level'], prefix='shop_review_num')
merge9 = pd.get_dummies(data['user_age_level'], prefix='user_age')
merge10 = pd.get_dummies(data['user_gender_id'], prefix='user_gender')
merge11 = pd.get_dummies(data['user_occupation_id'], prefix='user_occupation')
merge12 = pd.get_dummies(data['user_star_level'], prefix='user_star')
merge13 = pd.get_dummies(data['shop_star_level'], prefix='shop_star')
data['shop_review_positive_rate_nan'] = data.apply(lambda x:1 if x['shop_review_positive_rate'] == -1 else 0, axis=1)
data['shop_score_service_nan'] = data.apply(lambda x:1 if x['shop_score_service'] == -1 else 0, axis=1)
data['shop_score_delivery_nan'] = data.apply(lambda x:1 if x['shop_score_delivery'] == -1 else 0, axis=1)
data['shop_score_description_nan'] = data.apply(lambda x:1 if x['shop_score_description'] == -1 else 0, axis=1)
data = pd.concat([data, merge1, merge2, merge3, merge4, merge5, merge6, merge7,
                  merge8, merge9, merge10, merge11, merge12, merge13, merge_category], axis=1)
del merge1, merge2, merge3, merge4, merge5, merge6, merge7, merge8, merge9, merge10, merge11, merge12, merge_category
#data = data.drop(['item_id', 'user_id', 'context_id', 'shop_id', 'item_city_id','item_brand_id', 'item_city_id',
#                  'item_brand_id', 'context_page_id', 'item_collected_level', 'item_pv_level','item_pv_level',
#                  'item_price_level', 'item_sales_level', 'shop_review_num_level', 'user_age_level','user_gender_id',
#                  'user_occupation_id', 'user_star_level', 'shop_star_level'], axis=1)
for i in ['item_id', 'user_id', 'context_id', 'shop_id', 'item_city_id','item_brand_id', 'item_city_id','item_brand_id', 'context_page_id', 'item_collected_level', 'item_pv_level','item_pv_level','item_price_level', 'item_sales_level', 'shop_review_num_level', 'user_age_level','user_gender_id','user_occupation_id', 'user_star_level', 'shop_star_level']:
    try:
        del data[i]
    except:
        pass

#data = data.astype(float)

print('merge state 1 completed')

#hua chuang data
data.to_csv(path_or_buf=output_path+'data_handled.csv', index=None)
for i in range(day_min, day_max-1):
    data_temp = data[(data['context_timestamp_day'] < (i+3)) & (data['context_timestamp_day'] >= i)]
    name = 'train' + str(i) + '.csv'
    data_temp.to_csv(path_or_buf=output_path+name, index=None)

del data['predict_category_property'], data['item_property_list']

print('output data completed')

if go_pred == 1:
    train = data[~data['is_trade'].isnull()]
    test = data[data['is_trade'].isnull()]
    test_instance_id = test['instance_id']
    train_label = train['is_trade'].astype(int)
    del train['instance_id'], train['is_trade']
    del test['instance_id'], test['is_trade']
    train = train.fillna(0)
    test = test.fillna(0)
    
    del data
    
    print ('train test handle completed')
    
    #LR
    LR = LogisticRegression(C=1, solver='liblinear', max_iter=2000)
    LR.fit(train, train_label)
    predict_LR = LR.predict_proba(test)
    predict_LR = pd.DataFrame(predict_LR[:, 1])
    
    print ('LR completed')


'''
    #XGB
    predict_XGB = []
    kf = KFold(len(train), n_folds = 10, shuffle=True, random_state=5201314)
    for i, (train_index, test_index) in enumerate(kf):
        xgb_train = xgb.DMatrix(train.iloc[test_index], label=train_label.iloc[test_index])
        xgb_test = xgb.DMatrix(test)
        watchlist = [(xgb_train, 'train')]
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'lambda': 0,
            #'gamma': 5,
            'slient': 1,
            'alpha': 1,
            'max_depth': 20,
            #'subsample': 0.7,
            #'colsample_bytree': 0.7,
            'eta': 0.01,
            #'seed': 1,
            'n_jobs': 16}
        XGB = xgb.train(params,xgb_train,num_boost_round=20,evals=watchlist,verbose_eval=2,early_stopping_rounds=50)
        #XGB.fit(train.iloc[test_index], y=train_label.iloc[test_index], eval_metric='logloss')
        predict_XGB.append(XGB.predict(xgb_test))
        
        print (i, 'round completed')
    
    print ('XGB completed')
'''
'''
del train['context_id'], test['context_id']
k = 0
j = list(train.item_category_list_1.drop_duplicates())
for i in test.item_category_list_1.drop_duplicates():
    if i in j:
        k += 1
print (len(test.item_category_list_1.drop_duplicates()))

item_property_list_test = []
for i in range(len(test)):
    each_i = [x for x in test.ix[i, 'item_property_list'].split(';')]
    for j in each_i:
        item_property_list_test.append(j)
item_property_list_test = list(set(item_property_list_test))
'''