import numpy as np

import torch
import torch.nn as nn
from scipy import io
from tqdm import tqdm
from utils import sliding_window, count_sliding_window, grouper, metrics


def train(cnn, LEARNING_RATE, EPOCH, train_loader, hsi_train_patch, hsi_test_patch, lidar_train_patch, lidar_test_patch,
          hsi_train_label, hsi_test_label):
    print('The structure of the designed network', cnn)

    # move model to GPU
    cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)    # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    BestAcc = 0

    torch.cuda.synchronize()
    # train and test the designed model
    for epoch in range(EPOCH):
        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

            # move train data to GPU
            b_x1 = b_x1.cuda()
            b_x2 = b_x2.cuda()
            b_y = b_y.cuda()

            out1, out2, out3 = cnn(b_x1, b_x2)
            loss1 = loss_func(out1, b_y)
            loss2 = loss_func(out2, b_y)
            loss3 = loss_func(out3, b_y)

            loss = 0.01*loss1 + 0.01*loss2 + 1*loss3  # tune these hyperparameters to find an optimal result

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                cnn.eval()

                temp1 = hsi_train_patch
                temp1 = temp1.cuda()
                temp2 = lidar_train_patch
                temp2 = temp2.cuda()

                temp3, temp4, temp5 = cnn(temp1, temp2)

                pred_y1 = torch.max(temp3, 1)[1].squeeze()
                pred_y1 = pred_y1.cpu()
                # acc1 = torch.sum(pred_y1 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

                pred_y2 = torch.max(temp4, 1)[1].squeeze()
                pred_y2 = pred_y2.cpu()
                # acc2 = torch.sum(pred_y2 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

                pred_y3 = torch.max(temp5, 1)[1].squeeze()
                pred_y3 = pred_y3.cpu()
                # acc3 = torch.sum(pred_y3 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

                # weights are determined by each class accuracy
                Classes = np.unique(hsi_train_label)
                w0 = np.empty(len(Classes),dtype='float32')
                w1 = np.empty(len(Classes),dtype='float32')
                w2 = np.empty(len(Classes),dtype='float32')

                for i in range(len(Classes)):
                    cla = Classes[i]
                    right1 = 0
                    right2 = 0
                    right3 = 0

                    for j in range(len(hsi_train_label)):
                        if hsi_train_label[j] == cla and pred_y1[j] == cla:
                            right1 += 1
                        if hsi_train_label[j] == cla and pred_y2[j] == cla:
                            right2 += 1
                        if hsi_train_label[j] == cla and pred_y3[j] == cla:
                            right3 += 1

                    w0[i] = right1.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w1[i] = right2.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w2[i] = right3.__float__() / (right1 + right2 + right3 + 0.00001).__float__()

                w0 = torch.from_numpy(w0).cuda()
                w1 = torch.from_numpy(w1).cuda()
                w2 = torch.from_numpy(w2).cuda()

                pred_y = np.empty((len(hsi_test_label)), dtype='float32')
                number = len(hsi_test_label) // 5000
                for i in range(number):
                    temp = hsi_test_patch[i * 5000:(i + 1) * 5000, :, :, :]
                    temp = temp.cuda()
                    temp1 = lidar_test_patch[i * 5000:(i + 1) * 5000, :, :, :]
                    temp1 = temp1.cuda()
                    temp2 = w2*cnn(temp, temp1)[2] + w1*cnn(temp, temp1)[1] + w0*cnn(temp, temp1)[0]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                    del temp, temp2, temp3

                if (i + 1) * 5000 < len(hsi_test_label):
                    temp = hsi_test_patch[(i + 1) * 5000:len(hsi_test_label), :, :, :]
                    temp = temp.cuda()
                    temp1 = lidar_test_patch[(i + 1) * 5000:len(hsi_test_label), :, :, :]
                    temp1 = temp1.cuda()
                    temp2 = w2*cnn(temp, temp1)[2] + w1*cnn(temp, temp1)[1] + w0*cnn(temp, temp1)[0]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 5000:len(hsi_test_label)] = temp3.cpu()
                    del temp, temp2, temp3

                # pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == hsi_test_label).type(torch.FloatTensor) / hsi_test_label.size(0)

                io.savemat('./prediction.mat', {'prediction': pred_y})

                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy,
                       '| w0: %.2f' % w0[0], '| w1: %.2f' % w1[0], '| w2: %.2f' % w2[0])

                # save the parameters in network
                if accuracy > BestAcc:
                    torch.save(cnn.state_dict(), 'model.pkl')
                    BestAcc = accuracy
                    w0B = w0
                    w1B = w1
                    w2B = w2
                    weight0 = w0B.cpu().numpy()
                    weight1 = w1B.cpu().numpy()
                    weight2 = w2B.cpu().numpy()

                    io.savemat('./weights/w0_aspp_trento.mat', {'w0': weight0})
                    io.savemat('./weights/w1_aspp_trento.mat', {'w1': weight1})
                    io.savemat('./weights/w2_aspp_trento.mat', {'w2': weight2})
                    
                cnn.train()
    return cnn


def inference(net, img1, img2):
    """
    Test a model on a specific image
    """
    w0 = io.loadmat('./weights/w0_aspp_trento.mat')
    w0 = w0['w0']
    w1 = io.loadmat('./weights/w0_aspp_trento.mat')
    w1 = w1['w1']
    w2 = io.loadmat('./weights/w0_aspp_trento.mat')
    w2 = w2['w2']
    net.eval()
    patch_size = 11
    center_pixel = True
    batch_size, device = 64, 'cuda:0'
    n_classes = 16

    kwargs = {
        "step": 1,
        "window_size": (11, 11),
    }
    probs = np.zeros(img1.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img1, img2, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img1, img2, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)

                data2 = [b[1][0, 0] for b in batch]
                data2 = np.copy(data2)
                data2 = torch.from_numpy(data2)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)

                data2 = [b[1] for b in batch]
#                 print(data.shape, data2[0].shape)
                data2 = np.copy(data2)
                data2 = data2.transpose(0, 3, 1, 2)
                data2 = torch.from_numpy(data2)
                # data2 = data2.unsqueeze(1)

            indices = [b[2:] for b in batch]
            data = data.to(device)
            data2 = data2.to(device)
            output = net(data, data2)
            
            if isinstance(output, tuple):  # For multiple outputs
#                 output = output[0]
                temp2 = w2 * net(data, data2)[2] + w1 * net(data, data2)[1] + w0 * net(data, data2)[0]
                output = torch.max(temp2, 1)[1].squeeze()
                output = output.to("cpu")
            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs





