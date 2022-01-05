import numpy as np
import torch
from scipy import io
from sklearn.decomposition import PCA
import itertools
import mat73
from sklearn.metrics import confusion_matrix
from main import LIST_OF_PARAMETERS


FM = LIST_OF_PARAMETERS['fm']
NC = LIST_OF_PARAMETERS['nc']
Classes = LIST_OF_PARAMETERS['Classes']
batchsize = LIST_OF_PARAMETERS['batchsize']
patchsize1 = LIST_OF_PARAMETERS['patchsize1']
patchsize2 = LIST_OF_PARAMETERS['patchsize2']


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return


def load_data(dataset):
    dataset_basepath = '/home/h0/prpa810b/mmrst/Multimodal-Remote-Sensing-Toolkit/'
    if dataset == "Houston2013":
        DataPath1 = dataset_basepath + 'Houston2013/HSI.mat'
        DataPath2 = dataset_basepath + 'Houston2013/LiDAR.mat'
        TRPath = dataset_basepath + 'Houston2013/TRLabel.mat'
        TSPath = dataset_basepath + 'Houston2013/TSLabel.mat'

        Data = io.loadmat(DataPath1)
        Data = Data['HSI']
        Data = Data.astype(np.float32)

        Data2 = io.loadmat(DataPath2)
        Data2 = Data2['LiDAR']
        Data2 = Data2.astype(np.float32)

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TrLabel']
        TsLabel = TsLabel['TsLabel']

    elif dataset == "Houston2018":
        DataPath1 = dataset_basepath + 'Houston2018/HU_2018.mat'
        DataPath2 = dataset_basepath + 'Houston2018/LiDAR_Rsz.mat'
        TRPath = dataset_basepath + 'Houston2018/TRLabel.mat'
        TSPath = dataset_basepath + 'Houston2018/TSLabel.mat'

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

        Data = mat73.loadmat(DataPath1)
        Data = Data['HU_2018']
        Data = Data.astype(np.float32)
        Data2 = mat73.loadmat(DataPath2)
        Data2 = Data2['LiDAR_Rsz']
        Data2 = Data2.astype(np.float32)

    elif dataset == "Trento":
        DataPath1 = 'Trento/HSI.mat'
        DataPath2 = 'Trento/LiDAR.mat'
        TRPath = 'Trento/TrLabel.mat'
        TSPath = 'Trento/TsLabel.mat'

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

        Data = mat73.loadmat(DataPath1)
        Data = Data['HU_2018']
        Data = Data.astype(np.float32)
        Data2 = mat73.loadmat(DataPath2)
        Data2 = Data2['LiDAR_Rsz']
        Data2 = Data2.astype(np.float32)

    else:
        print("incorrect selection")

    return Data, Data2, TrLabel, TsLabel


def normalize(hsi_data, lidar_data):
    [m, n, l] = hsi_data.shape
    for i in range(l):
        minimal = hsi_data[:, :, i].min()
        maximal = hsi_data[:, :, i].max()
        hsi_data[:, :, i] = (hsi_data[:, :, i] - minimal) / (maximal - minimal)

    minimal = lidar_data.min()
    maximal = lidar_data.max()
    lidar_data = (lidar_data - minimal) / (maximal - minimal)

    return hsi_data, lidar_data


def preprocess(hsi_data, lidar_data):
    # extract the principal components
    [m, n, l] = hsi_data.shape
    PC = np.reshape(hsi_data, (m * n, l))
    pca = PCA(n_components=NC, copy=True, whiten=False)
    PC = pca.fit_transform(PC)
    PC = np.reshape(PC, (m, n, NC))

    # boundary interpolation
    temp = PC[:, :, 0]
    pad_width = np.floor(patchsize1 / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x = np.empty((m2, n2, NC), dtype='float32')

    for i in range(NC):
        temp = PC[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2

    x2 = lidar_data
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = np.int(pad_width2)
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2

    return x, x2


def create_hsi_patches(TrLabel, TsLabel, x, patchsize1=11, NC=20, pad_width=5):
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = TrLabel[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    # io.savemat('./patches/TrainPatch.mat', {'TrainPatch': TrainPatch})
    # io.savemat('./patches/TrainPatch.mat', {'TestPatch': TestPatch})
    print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)

    return TrainPatch, TestPatch, TrainLabel, TestLabel


def create_lidar_patches(TrLabel, TsLabel, x2, input_channels=1, patchsize2=11, pad_width2=5):
    # construct the training and testing set of LiDAR
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, input_channels, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1),
                :]
        patch = np.reshape(patch, (patchsize2 * patchsize2, input_channels))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (input_channels, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        patchlabel2 = TrLabel[ind1[i], ind2[i]]
        TrainLabel2[i] = patchlabel2

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, input_channels, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1),
                :]
        patch = np.reshape(patch, (patchsize2 * patchsize2, input_channels))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (input_channels, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        patchlabel2 = TsLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)

    return TrainPatch2, TestPatch2, TrainLabel2, TestLabel2


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        #ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    #     n_classes = np.max(target)

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm
    #     print(cm)

    # Compute global accuracy
    total = np.sum(cm)
    lis = [cm[x][x] for x in range(len(cm))]
    accuracy = 0
    for i in lis:
        accuracy += i
    accuracy *= 100 / float(total)

    # logger.log({"accuracy": accuracy})
    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    # logger.log({"F1 scores": F1scores})
    results["F1 scores"] = F1scores

    # Compute precision for every class
    Precisions = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            Precision = 1. * cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            Precision = 0.
        Precisions[i] = Precision

    # logger.log({"Precisions": Precisions})
    results["Precisions"] = Precisions

    # Compute Average Accuracy (AA)
    AAs = []
    for i in range(len(cm)):
        try:
            recall = cm[i][i] / np.sum(cm[i, :])
            if np.isnan(recall):
                continue
        except ZeroDivisionError:
            recall = 0.
        AAs.append(recall)
    results['AA'] = np.mean(AAs)
    # logger.log({"Average accuracy": AAs})
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa
    # logger.log({"Kappa": kappa})

    return results


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.
    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# from utils import grouper, sliding_window, count_sliding_window, camel_to_snake
def sliding_window(image1, image2, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size
    """
    # slide a window across the image
    w, h = window_size
    W, H = image1.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].
    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/
    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image1[x:x + w, y:y + h], image2[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, top2, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, top2, step, window_size, with_data=False)
    count = 0
    for _ in sw:
        count += 1
    #     print(count)
    return count


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([165, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([0, 165, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 0, 165]) / 255.
        if item == 19:
            y[index] = np.array([165, 165, 0]) / 255.
        if item == 20:
            y[index] = np.array([165, 165, 165]) / 255.
    return y
