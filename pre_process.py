import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import c_file, b_file, c_index, b_index

if __name__ == '__main__':
    print('reading click data...')
    clicks = pd.read_csv(c_file, header=None, names=c_index)
    clicks.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
    print(clicks.head(20))

    print('reading buy data...')
    buys = pd.read_csv(b_file, header=None, names=b_index)
    buys.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
    print(buys.head(20))

    print('buys.nunique(): ' + str(buys.nunique()))
    print('clicks.nunique(): ' + str(clicks.nunique()))

    clicks['valid_session'] = clicks.session_id.map(clicks.groupby('session_id')['item_id'].size() > 2)
    clicks = clicks.loc[clicks.valid_session].drop('valid_session', axis=1)
    print('clicks.nunique(): ' + str(clicks.nunique()))

    # #randomly sample a couple of them
    sampled_session_id = np.random.choice(clicks.session_id.unique(), 1000000, replace=False)
    clicks = clicks.loc[clicks.session_id.isin(sampled_session_id)]
    print('clicks.nunique(): ' + str(clicks.nunique()))

    print('clicks.isna().sum(): ' + str(clicks.isna().sum()))

    # average length of session
    print('average length of session : ' + str(clicks.groupby('session_id')['item_id'].size))

    item_encoder = LabelEncoder()
    clicks['item_id'] = item_encoder.fit_transform(clicks.item_id)
    print('df.head(): ' + str(clicks.head()))

    clicks['label'] = clicks.session_id.isin(buys.session_id)
    print('df.head(): ' + str(clicks.head()))

    print('drop duplicate: ' + str(clicks.drop_duplicates('session_id')['label'].mean()))
