import threading
import json


from train import Train

# Base Parameters
base_url = 'http://localhost:3000'


f = open('prediction-strategies.json')
strategies = json.load(f)
f.close()
 
# Iterating through the json
# list
for strategy in strategies:
    _model_id = strategy['id']
    _name = strategy['name']
    _feature = strategy['feature']
    _symbol = strategy['symbol'] 
    _modelTemplate = 'default'
    _formatter = 'default'

    train = Train(_name, _model_id, _modelTemplate, _formatter, _feature, _symbol, base_url)
    threading.Thread(target=train.run, args=()).start()