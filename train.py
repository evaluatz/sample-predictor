
 
import requests
import pandas as pd
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from formatters import Formatters
from modelTemplates import ModelTemplates


class Train:
    def __init__(self, name, model_id, modelTemplate, formatter, feature, symbol, base_url):
        self.name = name
        self.model_id = model_id
        self.modelTemplate = getattr(ModelTemplates, modelTemplate) 
        self.formatter = getattr(Formatters, formatter) 
        self.feature = feature
        self.symbol = symbol
        self.base_url = base_url

        self.timesteps = 4
        self.pred_size = self.timesteps * 20
        self.url_historic = '{base}/historic/{symbol}'.format(base=self.base_url, symbol=self.symbol)

    def run(self):
        logging.info('Starting {name}'.format(name=self.name))
        r = requests.get(self.url_historic)
        jsonObj = r.json()
        historic_data = pd.DataFrame(jsonObj['data'], columns=jsonObj['headers']).set_index("id").astype(float)
        df = self.formatter(historic_data)
 
        value_norm, df_norm = Formatters.normData(df)
        df_norm['target'] = df_norm[self.feature]

        model = self.modelTemplate(df_norm, self.feature, self.timesteps, self.pred_size)
        logging.info('Saving Model {name}'.format(name=self.name))
        model.save('./models/{model_id}'.format(model_id=self.model_id ))
        logging.info('Finished {name}'.format(name=self.name))
        return df
