from fastapi import FastAPI, Path
from typing import Union, List, Dict
import tensorflow as tf
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import warnings
import os
import json
# import time
# import sys
# import traceback
from pydantic import BaseModel
import pickle
import helper


class HorseInfo(BaseModel):
    list_info: List[Dict]

    ...


data_path = './app_data/'
default_data = {'data': ''}

################
# Retrieve data
with open(data_path + 'init_val.json', 'r') as f:
    init_val = json.loads(f.read())

with open(data_path + 'selected_features.json', 'r') as f:
    features = json.loads(f.read())

with open(data_path + 'description.json', 'r') as f:
    examples = json.loads(f.read())


jockey_cols = features['jockey']
jockey_info_df = pd.read_csv(
    data_path+'jockey_info.csv', dtype={'KisyuCode': str})

coach_info_df = pd.read_csv(
    data_path+'coach_info.csv', dtype={'ChokyosiCode': str})

owner_info_df = pd.read_csv(
    data_path+'owner_info.csv', dtype={'BanusiCode': str})

historical_races = pd.read_csv(
    data_path+'historical_race_results.csv').astype({'KettoNum': str}).groupby('raceid')

# merge horse, owner, coach infos
horse_cols = features['horse']
horse_info_df = pd.read_csv(data_path+'horse_info.csv',
                            dtype={'KettoNum': str, 'ChokyosiCode': str, 'BanusiCode': str})
horse_info_df = pd.merge(left=horse_info_df,
                         right=coach_info_df,
                         on='ChokyosiCode',
                         how='left')
horse_info_df = pd.merge(left=horse_info_df,
                         right=owner_info_df,
                         on='BanusiCode',
                         how='left')


################
# Retrieve model
model_nn = tf.keras.models.load_model('modelcheckpoint/model_nn.hdf5')
model_xgb = XGBClassifier()
model_xgb.load_model(os.path.join('modelcheckpoint', 'model_xgb.json'))
scaler = pickle.load(open(data_path+'scaler.pickle', 'rb'))


################
# Functions
def retrieve_race(raceid):
    raceid_str = raceid['date'] + ' ' + \
        raceid['race'] + ':' + raceid['trackcd']
    try:
        race_result = historical_races.get_group(raceid_str)
        race_result = race_result[['KettoNum',
                                   'Top 3 pred', 'Truth']].to_dict('records')
    except:
        race_result = default_data
    return race_result


def retrieve_infos(horses=None, jockeys=None, tracktype="Turf"):
    assert len(horses) == len(
        jockeys), f'Lenght of horses and jockeys must be equal, got {len(horses)} and {len(jockeys)}'
    assert tracktype in ["Dirt","Turf","Jump"], 'Tracktype must be "Dirt","Turf" or "Jump"'
    horses_in_race = pd.DataFrame(horses, columns=['KettoNum'])
    jockeys_in_race = pd.DataFrame(jockeys, columns=['KisyuCode'])
    horses_in_race = pd.merge(left=horses_in_race,
                              right=horse_info_df,
                              on='KettoNum',
                              how='left')
    for i in ["Dirt","Turf","Jump"]:
        horses_in_race["Track_"+i] = 0.
    horses_in_race["Track_"+tracktype] = 1.
    jockeys_in_race = pd.merge(left=jockeys_in_race,
                               right=jockey_info_df,
                               on='KisyuCode',
                               how='left')

    return pd.merge(left=horses_in_race,
                    right=jockeys_in_race,
                    left_index=True,
                    right_index=True,
                    how='left').set_index('KettoNum')


def process_missing_data(df: pd.DataFrame):
    for col in features['selected_features']:
        df.fillna(init_val[col], inplace=True)


def process_custom_info(infos, df: pd.DataFrame):
    for horse in infos:
        if horse['KettoNum'] not in df.index:
            continue
        for info in list(horse.keys())[1:]:
            if info not in ["SexCD", "HinsyuCD", "TozaiCD"]:
                key = info
                val = horse[info]
            else:
                if isinstance(horse[info], str):
                    info + '_' + horse[info]
                else:
                    key = info + '_' + str(int(horse[info]))
                val = 1.
            try:
                df.loc[horse["KettoNum"], key] = val
            except:
                warnings.warn(
                    f"Unable to set {info} of {horse['KettoNum']} into data frame")


def preprocess(horses, jockeys, horseinfos, tracktype):
 
    # integrate data
    race_df = retrieve_infos(horses=horses, jockeys=jockeys, tracktype=tracktype)

    # deal with missing data
    if horseinfos is not None:
        process_custom_info(horseinfos, race_df)
    process_missing_data(race_df)
    # scale data
    # data = scaler.transform(race_df[features['selected_features']])
    data = race_df[features['selected_features']]

    return data, race_df


def predict(data, models):
    pred = helper.predict_top3_prob_multimodels(
        models=models, Xs=data, raceid=np.ones(shape=(len(data[0]),)))
    pred.drop(['raceid', 'Top 3 prob'], axis=1, inplace=True)
    return pred


def postprocess(pred: pd.DataFrame, horses):
    pred['KettoNum'] = horses
    return pred[['KettoNum', 'Top 3 pred']].to_dict(orient='records')


#######
# API
app = FastAPI(description="""
descriptions: descriptions of predictor's parameters with examples

predictor: predict top 3 horse of race with provided input""")


@app.get("/")
def root_page():
    return {"title": "jra horse racing prediction"}


@app.get("/descriptions/{code}")
def description(code:str):
    '''
    Descriptions of predictor's parameters with examples
    
    Valid codes: SexCD, HinsyuCD, TozaiCD, raceid, horses, jockeys, horseinfos, tracktype'''
    if code in examples:
        return examples[code]
    else:
        return default_data

@app.post("/predictor/")
def prediction(raceid: Union[Dict, None] = None,
               horses: Union[List[str], None]= None,
               jockeys: Union[List[str], None] = None,
               tracktype: str = "Turf",
               horseinfos: Union[List, None] = None):
    """
    If raceid is provided, then the predictions (and the results for the histotical races) will be returned,

    Otherwise, horses and jockeys (with the same lengths) must be provided.

    Note: tracktype is optional, default: Turf

    horseinfos is optional

    Go to /descriptions/{code} for descriptions and examples
    """

    if raceid is None and horses is None:
        return {'data': ''}
    try:
        if raceid:
            return retrieve_race(raceid)

        else:
            data, race_df = preprocess(horses=horses, jockeys=jockeys, horseinfos=horseinfos, tracktype=tracktype)
            result = predict(data=[data,scaler.transform(data)], models=[model_xgb, model_nn])
            result = postprocess(result, horses)
            return result
    except Exception as e:
        return {'error': str(e)}
