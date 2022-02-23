import pandas as pd


class Formatters:
    def normData(_df):
        value_norm = _df.std().max()
        df_norm = _df / value_norm
        df_norm = df_norm.where(df_norm <= 1, 1)
        df_norm = df_norm.where(df_norm >= -1, -1)
        return value_norm, df_norm

    def default(df):
        s_high = ((df['high']- df['open']) / df['open']).rename("high") 
        s_low = ((df['low']- df['open']) / df['open']).rename("low") 
        s_ma50 = ((df['ma50']- df['open']) / df['open']).rename("ma50")
        s_ma100 = ((df['ma100']- df['open']) / df['open']).rename("ma100")
        s_ma200 = ((df['ma200']- df['open']) / df['open']).rename("ma200")

        s_rsi_c = (df['rsi14']-30)/(df['rsi14']-30).std()
        s_rsi = s_rsi_c.where(s_rsi_c <= 1, 1).where(s_rsi_c >= -1, -1)

        return pd.concat([s_high, s_low, s_ma50, s_ma100, s_ma200, s_rsi],axis=1)