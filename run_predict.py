import threading
import json
import schedule
import time

from predict import Predict

# Base Parameters
base_url = 'https://api.evaluatz.com'


f = open('prediction-strategies.json')
strategies = json.load(f)
f.close()


def job():
    for strategy in strategies: 
        _model_id = strategy['id']
        _secret = strategy['secret']
        _name = strategy['name']
        _formatter = 'default'
        _symbol = strategy['symbol'] 

        predict = Predict(_name, _model_id, _secret,  _formatter, _symbol, base_url)
        threading.Thread(target=predict.run, args=()).start()

schedule.every().hour.at(":00").do(job)
schedule.every().hour.at(":15").do(job)
schedule.every().hour.at(":30").do(job)
schedule.every().hour.at(":45").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)