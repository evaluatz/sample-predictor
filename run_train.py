import threading
import json


from train import Train

# Base Parameters
base_url = 'https://api.evaluatz.com'
#base_url = 'http://200.98.82.81'


f = open('prediction-strategies.json')
strategies = json.load(f)
f.close()

for strategy in strategies:
    _model_id = strategy['id']
    _name = strategy['name']
    _feature = strategy['feature']
    _symbol = strategy['symbol'] 
    _modelTemplate = 'default'
    _formatter = 'default'

    train = Train(_name, _model_id, _modelTemplate, _formatter, _feature, _symbol, base_url)
    #train.run()
    threading.Thread(target=train.run, args=()).start()