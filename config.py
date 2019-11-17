import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

embed_dim = 128
batch_size = 1024
image_folder = 'data'
buy_data_file = 'data/yoochoose-data/yoochoose-buys.dat'
click_data_file = 'data/yoochoose-data/yoochoose-clicks.dat'
test_data_file = 'data/yoochoose-data/yoochoose-test.dat'

# Training parameters
num_workers = 4  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 10  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
