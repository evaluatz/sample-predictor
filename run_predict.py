import json
import subprocess

# Base Parameters
base_url = 'https://api.evaluatz.com'
#base_url = 'http://200.98.82.81'


f = open('prediction-strategies.json')
strategies = json.load(f)
f.close()

processes = []

for strategy in strategies: 
    _model_id = strategy['id']
    _secret = strategy['secret']
    _name = strategy['name']
    _formatter = 'default'
    _symbol = strategy['symbol'] 

    processExec = ["C:\Python310\python.exe", "C:\\Users\\guign\\Documents\\00-Projects\\sample-predictor\\execute_predict.py"]
    processExec.extend(["-n", _name])
    processExec.extend(["-m", _model_id])
    processExec.extend(["-x", _secret])
    processExec.extend(["-f", _formatter])
    processExec.extend(["-s", _symbol])
    processExec.extend([ "-b", base_url])
    p = subprocess.Popen(processExec)
    processes.append(p)


for p in processes:
    p.wait()