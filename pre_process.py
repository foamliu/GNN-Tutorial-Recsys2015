import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import click_data_file, buy_data_file

if __name__ == '__main__':
    # print('reading click data...')
    # df = pd.read_csv(click_data_file, header=None)
    # df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
    # print(df.head(20))

    print('reading buy data...')
    buy_df = pd.read_csv(buy_data_file, header=None)
    buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
    print(buy_df.head(20))

    print('buy_df.nunique(): ' + str(buy_df.nunique()))
    # print('df.nunique(): ' + str(df.nunique()))
    #
    # df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
    # df = df.loc[df.valid_session].drop('valid_session', axis=1)
    # print('df.nunique(): ' + str(df.nunique()))

    # # #randomly sample a couple of them
    # sampled_session_id = np.random.choice(df.session_id.unique(), 1000000, replace=False)
    # df = df.loc[df.session_id.isin(sampled_session_id)]
    # print('df.nunique(): ' + str(df.nunique()))
    #
    # print('df.isna().sum(): ' + str(df.isna().sum()))
    #
    # # average length of session
    # print('average length of session : ' + str(df.groupby('session_id')['item_id'].size))
    #
    # item_encoder = LabelEncoder()
    # df['item_id'] = item_encoder.fit_transform(df.item_id)
    # print('df.head(): ' + str(df.head()))
    #
    # df['label'] = df.session_id.isin(buy_df.session_id)
    # print('df.head(): ' + str(df.head()))
    #
    # print('drop duplicate: ' + str(df.drop_duplicates('session_id')['label'].mean()))
