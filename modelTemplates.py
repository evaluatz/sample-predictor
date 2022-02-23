
import os
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


class ModelTemplates:
    def default( df_, field_target, timesteps, pred_size):
        df_f = df_.copy()
        df_f['target'] = df_f[field_target].shift(-1)
        df_f.dropna(inplace=True)
        df_f_scaler = df_f.values
        df_f_rows = df_f.shape[0]

        #separate in model an predictions
        df_model = df_f_scaler[df_f_rows % timesteps:df_f_rows - pred_size]

        test_size = df_model.shape[0] - pred_size

        df_train = df_model[:,0:df_model[0].size-1][:test_size]
        df_train_target = df_model[:,df_model[0].size-1][:test_size]

        df_cross_validation = df_model[:,0:df_model[0].size-1][test_size:]
        df_cross_validation_target = df_model[:,df_model[0].size-1][test_size:]

            ##############
            #reshape 
            #########
        df_train =  np.reshape(df_train, (round(df_train.shape[0]/timesteps), df_train.shape[1], timesteps))
        df_train_target = np.reshape(df_train_target, (round(df_train_target.shape[0]/timesteps), timesteps))

        df_cross_validation =  np.reshape(df_cross_validation, (round(df_cross_validation.shape[0]/timesteps), df_cross_validation.shape[1], timesteps))
        df_cross_validation_target = np.reshape(df_cross_validation_target, (round(df_cross_validation_target.shape[0]/timesteps), timesteps))

            ####################/
            ## CREATE MODEL ##/
            ################/
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(100, return_sequences=True))
        model.add(keras.layers.LSTM(100, return_sequences=False))
        #     model.add(keras.layers.Dense(256, activation='elu'))
        #     model.add(keras.layers.Dense(256, activation='selu'))
        model.add(keras.layers.Dense(25))
        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
        )

        model.fit(
            x=df_train, 
            y=df_train_target,
            epochs=20,
            verbose=0,
            validation_data=(df_cross_validation, df_cross_validation_target),
        )

        return model