import os
import errno
import requests
import pandas as pd
import numpy as np
import math
import json
import time
from datetime import datetime
from datetime import timedelta
import dateutil.parser
import tensorflow as tf
import logging


from formatters import Formatters


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: 
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

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
        self.url_historic = '{base}/historic/{symbol}?offset=1000'.format(base=base_url, symbol=self.symbol)
        self.url_update = '{base}/prediction'.format(base=base_url)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now()
        last_update = current_time - timedelta(minutes=current_time.minute % 15) - timedelta(seconds=current_time.second)
        foldername = '{current_dir}\logs\{lst_update}'.format(current_dir=current_dir, lst_update=last_update.strftime("%Y%m%d_%H_%M_%S"))
        mkdir_p(foldername)

        filename = '{model_id}_{symbol}'.format(symbol=symbol,model_id=model_id) 
        log_path = '{foldername}\{filename}.log'.format(foldername=foldername,filename=filename) 
        print("log location", log_path)
        logging.basicConfig(filename=log_path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    def run(self):
        with tf.device('/CPU:0'):
            logging.info('Starting prediction {name}'.format(name=self.name))
            model = tf.keras.models.load_model('./models/{model_id}'.format(model_id=self.model_id))
            logging.info('Model loaded {name}'.format(name=self.name))

            startTime = time.time()-720
            is_updating = True

            symb_details_res = requests.get(self.url_symbol_details)
            while(startTime >= dateutil.parser.isoparse(symb_details_res.json()['lastUpdate']).timestamp()):
                logging.info('Waiting to update {name}'.format(name=self.name))
                time.sleep(5)
                symb_details_res = requests.get(self.url_symbol_details)

            logging.info('Loading data {name}'.format(name=self.name))
            r = requests.get(self.url_historic)
            logging.info('Doing the magic {name}'.format(name=self.name))
            jsonObj = r.json()
            historic_data = pd.DataFrame(jsonObj['data'], columns=jsonObj['headers']).set_index("id").astype(float)
            df = self.formatter(historic_data)
            value_norm, df_norm = Formatters.normData(df)


            pred_raw_df = df_norm.values  
            pred_raw_df = pred_raw_df[pred_raw_df.shape[0] % self.timesteps:pred_raw_df.shape[0]]
            pred_np_df = np.reshape(pred_raw_df, (round(pred_raw_df.shape[0]/self.timesteps), pred_raw_df.shape[1], self.timesteps))

            pred_low_val_df = model.predict(pred_np_df) * value_norm
            df_pred = historic_data.tail(historic_data.shape[0] - (historic_data.shape[0] % self.timesteps)).reset_index()
            df_pred['pred'] = range(0, df_pred.shape[0])
            df_pred['pred'] = df_pred['pred'].apply(lambda x: pred_low_val_df[:,0][math.floor(x/self.timesteps)])
            df_pred['pred_value'] = (df_pred['pred']+1)  * df_pred['open'] 
            df_pred_last = df_pred.tail(1)[['id','open', 'pred_value' ]]

            data = json.dumps({ 
                "openTime": df_pred_last['id'].values[0],
                "strategyID": self.model_id,
                "secret": self.secret,
                "value": round(df_pred_last['pred_value'].values[0],2)
            })
            logging.info('Saving prediction {name}'.format(name=self.name))

            max_tries = 10
            current_try = 0
            is_updating = True
            update_res = None
            while is_updating and max_tries > current_try:
                
                try:
                    current_try = current_try + 1
                    logging.info('Saving prediction {data}'.format(data=data))
                    headers = {
                        'Content-type': 'application/json',
                         "User-Agent": "Mozilla/5.0 (Linux; Android 7.0.0; SM-G960F Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36"
                         }
                    update_res = requests.post(self.url_update, data = data, headers=headers)
                    if update_res.status_code == 201:
                        is_updating = False
                    else:
                        logging.error('Saving prediction {update_res}'.format(update_res=str(update_res.json())))
                except:
                    logging.error('Saving prediction {name}'.format(name=self.name, update_res=str(update_res)))
                time.sleep(5)
            logging.info('Finished prediction {name}'.format(name=self.name))
            return data