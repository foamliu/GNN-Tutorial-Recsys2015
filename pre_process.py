import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import c_file, b_file, c_index, b_index
from utils import get_logger

if __name__ == '__main__':
    logger = get_logger()

    logger.info('reading click data...')
    clicks = pd.read_csv(c_file, header=None, names=c_index, low_memory=False)
    logger.info(clicks.head(20))

    logger.info('reading buy data...')
    buys = pd.read_csv(b_file, header=None, names=b_index)
    logger.info(buys.head(20))

    logger.info('buys.nunique(): ' + str(buys.nunique()))
    logger.info('clicks.nunique(): ' + str(clicks.nunique()))

    clicks['valid_session'] = clicks.session_id.map(clicks.groupby('session_id')['item_id'].size() > 2)
    clicks = clicks.loc[clicks.valid_session].drop('valid_session', axis=1)
    logger.info('clicks.nunique(): ' + str(clicks.nunique()))

    # #randomly sample a couple of them
    sampled_session_id = np.random.choice(clicks.session_id.unique(), 1000000, replace=False)
    clicks = clicks.loc[clicks.session_id.isin(sampled_session_id)]
    logger.info('clicks.nunique(): ' + str(clicks.nunique()))

    logger.info('clicks.isna().sum(): ' + str(clicks.isna().sum()))

    # average length of session
    logger.info('average length of session : ' + str(clicks.groupby('session_id')['item_id'].size))

    item_encoder = LabelEncoder()
    clicks['item_id'] = item_encoder.fit_transform(clicks.item_id)
    logger.info('df.head(): ' + str(clicks.head()))

    clicks['label'] = clicks.session_id.isin(buys.session_id)
    logger.info('df.head(): ' + str(clicks.head()))

    logger.info('drop duplicate: ' + str(clicks.drop_duplicates('session_id')['label'].mean()))
