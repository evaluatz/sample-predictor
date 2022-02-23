import requests
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
import dateutil.parser
import tensorflow as tf
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
from formatters import Formatters


class Predict:
    def __init__(self,name, model_id, secret,  formatter, symbol, base_url):
        self.name = name
        self.model_id = model_id
        self.secret = secret
        self.formatter = getattr(Formatters, formatter) 
        self.symbol = symbol
        self.timesteps = 4
        self.pred_size = self.timesteps * 20
        self.url_symbol_details = '{base}/symbol/{symbol}'.format(base=base_url, symbol=self.symbol)
        self.url_historic = '{base}/historic/{symbol}'.format(base=base_url, symbol=self.symbol)
        self.url_update = '{base}/prediction'.format(base=base_url)

    def run(self):
        logging.info('Starting prediction {name}'.format(name=self.name))
        startTime = time.time()-60
        while(startTime >= dateutil.parser.isoparse(requests.get(self.url_symbol_details).json()['lastUpdate']).timestamp()):
            logging.info('Waiting to update {name}'.format(name=self.name))
            time.sleep(1)

        r = requests.get(self.url_historic)
        jsonObj = r.json()
        historic_data = pd.DataFrame(jsonObj['data'], columns=jsonObj['headers']).set_index("id").astype(float)
        df = self.formatter(historic_data)
        value_norm, df_norm = Formatters.normData(df)

        model = tf.keras.models.load_model('./models/{model_id}'.format(model_id=self.model_id))

        pred_raw_df = df_norm.values  
        pred_raw_df = pred_raw_df[pred_raw_df.shape[0] % self.timesteps:pred_raw_df.shape[0]]
        pred_np_df = np.reshape(pred_raw_df, (round(pred_raw_df.shape[0]/self.timesteps), pred_raw_df.shape[1], self.timesteps))

        pred_low_val_df = model.predict(pred_np_df) * value_norm
        df_pred = historic_data.tail(historic_data.shape[0] - (historic_data.shape[0] % self.timesteps)).reset_index()
        df_pred['pred'] = range(0, df_pred.shape[0])
        df_pred['pred'] = df_pred['pred'].apply(lambda x: pred_low_val_df[:,0][math.floor(x/self.timesteps)])
        df_pred['pred_value'] = (df_pred['pred']+1)  * df_pred['open'] 
        df_pred_last = df_pred.tail(1)[['id','open', 'pred_value' ]]

        data = { 
            "openTime": df_pred_last['id'].values[0],
            "strategyID": self.model_id,
            "secret": self.secret,
            "value": round(df_pred_last['pred_value'].values[0],2)
        }
        logging.info('Saving prediction {name}'.format(name=self.name))
        update = requests.post(self.url_update, json = data)
        logging.info('Finished prediction {name}'.format(name=self.name))
        return data