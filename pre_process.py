import numpy as np
import pandas as pd
import torch
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

    item_encoder = LabelEncoder()
    clicks['item_id'] = item_encoder.fit_transform(clicks.item_id)
    logger.info(clicks.head())

    # randomly sample a couple of them
    sampled_session_id = np.random.choice(clicks.session_id.unique(), 1000000, replace=False)
    clicks = clicks.loc[clicks.session_id.isin(sampled_session_id)]
    logger.info(clicks.nunique())

    clicks['label'] = clicks.session_id.isin(buys.session_id)
    logger.info(clicks.head())

    processed_path = 'data/yoochoose_click_binary_1M_sess.dataset'
    with open(processed_path, 'wb') as f:
        torch.save(clicks, f)
