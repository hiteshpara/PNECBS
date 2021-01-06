# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import requests
import pandas as pd
import numpy as np
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors


def cont_res_ret():
    api_url = "http://127.0.0.1:8000/api/content-detail/"
    content_id = "http://127.0.0.1:8000/api/content-id/"
        
    resid = requests.get(content_id).json()
    cid = resid['itemid']
       
    content_id = str(cid)
    URL = api_url+content_id
    response = requests.get(URL).json() 
    return response

def item_res_ret():
    api_url = "http://127.0.0.1:8000/api/item-detail/"
    item_id = "http://127.0.0.1:8000/api/content-id/"
        
    resid = requests.get(item_id).json()
    cid = resid['itemid']
       
    content_id = str(cid)
    URL = api_url+content_id
    response = requests.get(URL).json() 
    return response

def knn():
    df = pd.read_csv('media/demo.csv')
    df = df.replace("Not_found",np.NaN)
    nu_F = df.iloc[:,3:].columns
    for i in nu_F:
        df[i] = df[i].astype(str).astype(float)

    ls = []
    for i in df['S.no']:
        req_q = df[df['S.no']==i].drop(df.columns[df.apply(lambda col: col.isnull().sum()>=1)], axis=1)
        req_q = req_q.drop(['S.no','item','Description'],axis=1)
        mode_req = req_q.mode(axis=1)
        ls.append(mode_req[0].values[0])
    df['Mode'] = ls
    df.sort_values(by=['Mode'])
    df = df.drop(['S.no'],axis=1)

    df['item_copy'] = df['item']
    df['Description_copy'] = df['Description']

    labels_ordered=df.groupby(['Description'])['Mode'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df['Description']=df['Description'].map(labels_ordered)

    temp = {}
    for i,j in enumerate(df['item']):
        temp.setdefault(j,i)
    L = []
    for i,j in temp.items():
        L.append(i)

    dic = {}
    for i in range(0,len(L)):
        dic.setdefault(L[i],i)
    df['item']=df['item'].map(dic)

    dataset = df.drop(['Cost_Amazon','Cost_MI','Cost_flipkart','Cost_RelianceDigital','Cost_Apple'],axis=1)
    req_df = dataset.iloc[:,0:3]
    X_col = req_df.iloc[:,0:2].columns
    X = req_df[X_col].values
    Y = req_df['Mode'].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(x_train_scaled)
    x_test_scaled = scaler.fit_transform(X_test)
    X_test = pd.DataFrame(x_test_scaled)

    params = {'n_neighbors':[1,2]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=3)
    model.fit(X_train,Y_train)
    req_K = model.best_params_['n_neighbors']

    model = neighbors.KNeighborsRegressor(n_neighbors = req_K)
    model.fit(X_train, Y_train) 

    return [model,df]

class ActionCI(Action):

    def name(self) -> Text:
        return "action_cost"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        ent = tracker.latest_message['entities']
        
        response = cont_res_ret()

        for e in ent:
            if e['entity'] == 'info':
                val = e['value']
                if val=='cost' or val == 'price':
                    msg = str(response['cost'])
                else:
                    msg="No details"    
        dispatcher.utter_message(text=msg)

        return []

class ActionRec(Action):

    def name(self) -> Text:
        return "action_recommendn"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        ent = tracker.latest_message['entities']     
        model = knn()[0]
        df = knn()[1]
        response_it = item_res_ret()
        response_con = cont_res_ret()    

        it = response_it['title']
        ds = response_con['specs']
        q_df = df[df['item_copy'].str.contains(it[:-1]) & df['Description_copy'].str.contains(ds)]
        if q_df is not None:
            i = q_df['item'].values
            j = q_df['Description'].values
            x = pd.DataFrame([[i[0],j[0]]])
            print(x)
            pred = str(int(model.predict(x)[0]))
        else:
            pred = "Not Found any records"  

        dispatcher.utter_message(text=pred)
        if int(pred) > response_con['cost']:
            dispatcher.utter_message(text="PNECBS is showing the best price")

        return []


