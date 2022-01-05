import argparse
import torch
import numpy as np
import torch.utils.data as dataf
from scipy import io
# import matplotlib.pyplot as plt

from utils import load_data, get_device, normalize, preprocess, create_hsi_patches, create_lidar_patches, metrics, \
                    list_to_colormap
from model import ASPP_CNN
from model_utils import train, inference


dataset_names = ["Houston2013", "Houston2018", "Trento"]
# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs (default: 100)")

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Set random seed",
)

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
DATASET = args.dataset
EPOCH = args.epoch
LEARNING_RATE = 0.001
NUMBER_OF_COMPONENTS = 20
PATCHSIZE_1 = 11
PATCHSIZE_2 = 11
BATCHSIZE = 64
PAD_WIDTH=5
FEATURE_MAPS = 32
if not DATASET == "HOUSTON2018":
    input_channels = 1
else:
    input_channels = 17

LIST_OF_PARAMETERS = {"dataset":DATASET, "epoch":EPOCH, "learning_rate":LEARNING_RATE, "nc":NUMBER_OF_COMPONENTS,
                      "patchsize1":PATCHSIZE_1, "patchsize2":PATCHSIZE_2, "batchsize":BATCHSIZE, "padwidth": PAD_WIDTH,
                      "fm":FEATURE_MAPS, "ip_channel":input_channels}


# step 1: load dataset and perform normalization and preprocessing
hsi_data, lidar_data, tr_label, ts_label = load_data(DATASET)
hsi_data, lidar_data = normalize(hsi_data, lidar_data)
hsi_data_processed, lidar_data_processed = preprocess(hsi_data, lidar_data)
hsi_train_patch, hsi_test_patch, hsi_train_label, hsi_test_label = create_hsi_patches(tr_label, ts_label, hsi_data_processed, PATCHSIZE_1, NUMBER_OF_COMPONENTS, PAD_WIDTH)
lidar_train_patch, lidar_test_patch, lidar_train_label, lidar_test_label = create_lidar_patches(tr_label, ts_label, lidar_data_processed, input_channels, PATCHSIZE_2, PAD_WIDTH)

# step 2: change data to the input type of PyTorch
hsi_train_patch = torch.from_numpy(hsi_train_patch)
hsi_train_label = torch.from_numpy(hsi_train_label)-1
hsi_train_label = hsi_train_label.long()

hsi_test_patch = torch.from_numpy(hsi_test_patch)
hsi_test_label = torch.from_numpy(hsi_test_label)-1
hsi_test_label = hsi_test_label.long()

Classes = len(np.unique(hsi_train_label))
LIST_OF_PARAMETERS['classes'] = Classes

lidar_train_patch = torch.from_numpy(lidar_train_patch)
lidar_train_label = torch.from_numpy(lidar_train_label)-1
lidar_train_label = lidar_train_label.long()

lidar_test_patch = torch.from_numpy(lidar_test_patch)
lidar_test_label = torch.from_numpy(lidar_test_label)-1
lidar_test_label = lidar_test_label.long()

dataset = dataf.TensorDataset(hsi_train_patch, lidar_train_patch, lidar_train_label)
train_loader = dataf.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

#step 3: training and generating metrics
net = ASPP_CNN()
net = train(net, LEARNING_RATE, EPOCH, train_loader, hsi_train_patch, hsi_test_patch, lidar_train_patch, lidar_test_patch,
          hsi_train_label, hsi_test_label)

prediction = io.loadmat('/prediction.mat')['prediction']
target = hsi_test_label
results = metrics(prediction, target)
print(results)

#step 4: Generate Classification Map
# probabilities = inference(net, img1, img2)

# [row,col,ht] = prediction.shape
# temp = np.zeros((prediction.shape[0], prediction.shape[1]))

# for i in range(row):
#     for j in range(col):
#         idxarr=[]
#         idxarr = np.nonzero(prediction[i][j])
#         if idxarr[0].size :
#             k=idxarr[0][0]
#             temp[i][j] = int(prediction[i][j][k])
#         else:
#             temp[i][j] = 0

# data = temp
# [m, n] = data.shape

# x = np.ravel(data)
# y_list = list_to_colormap(x)
# y_re = np.reshape(y_list, (m, n, 3))

# fig = plt.figure(frameon=False)
# # fig.set_size_inches(data.shape[1], data.shape[0])
# fig.set_size_inches(data.shape[1]*2.0/100, data.shape[0]*2.0/100)
# plt.axis('off')
# plt.imshow(y_re)
# plt.savefig('classification_maps/prediction_map')






