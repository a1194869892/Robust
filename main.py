import ot
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import next_batch
from load_datamat import get_datamat
from post_clustering import spectral_clustering, acc, nmi, ari, acc_with_mapping_shuffled, count_misclassifications, \
    count_misclassifications_with_confusion_matrix, best_map
import scipy.io as sio
import math
import warnings
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


class Conv2dSamePad(nn.Module):

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels, padding):

        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))
        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        padding = list(reversed(padding))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

            self.decoder.add_module('pad%d' % (i + 1), nn.ReflectionPad2d(padding[i]))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x, diag=True):  # shape=[n, d]
        if diag is True:
            y = torch.matmul(self.Coefficient, x)
        else:
            y = torch.matmul(self.Coefficient - torch.diag(torch.diag(self.Coefficient)), x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, padding, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels, padding)
        self.self_expression = SelfExpression(self.n)

    def reset_coef(self, coef):
        self.self_expression.Coefficient.data = coef

    def forward(self, x):
        z = self.ae.encoder(x)
        shape = z.shape
        z = z.view(shape[0], -1)
        z_recon = self.self_expression(z)
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)
        return x_recon, z, z_recon

    def compute_ot(self):
        cost = 1 - F.softmax(self.self_expression.Coefficient)
        mu = torch.ones(self.n)
        mu = mu / mu.sum()
        nu = torch.ones((self.n,))
        nu = nu / nu.sum()

        mu = mu.to(cost.device)
        nu = nu.to(cost.device)

        Wd = ot.emd2(mu, nu, cost)  # metrics 0.2....0.1
        Plan_mat = ot.sinkhorn(mu, nu, cost, 1.0, stopThr=1e-3)  # solution
        return Plan_mat

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp, weight_ot):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        P = self.compute_ot()
        loss_ot = -torch.sum(P * self.self_expression.Coefficient)
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp + weight_ot * loss_ot

        return loss


def pretrain_ae(model, optimizer, x, epochs, batch_size, pretrain_path, update=100):
    for epoch in range(epochs):
        cost = 0
        for batch_x, batch_no in next_batch(x, batch_size):
            batch_x_recon = model(batch_x)
            loss = F.mse_loss(batch_x_recon, batch_x, reduction='sum')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cost += loss.item()
        if epoch > int(epochs / 3) and epoch % update == 0:
            print('Epochs: %d, Loss: %.4f' % (epoch, cost))
            pretrain_path_temp = pretrain_path.split('.pkl')[0] + '_%d.pkl' % (int(epoch / update))
            torch.save(model.state_dict(), pretrain_path_temp)
    torch.save(model.state_dict(), pretrain_path)


def train(model,  # type: #DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, weight_ot=0.1, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                             weight_ot=weight_ot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch > 0 and epoch % 10 == 0:
            C = model.compute_ot().detach().to('cpu').numpy()
            y_pred, U = spectral_clustering(C, K, dim_subspace, alpha, ro)
            if epoch % 1 == 0 or epoch == 0:
                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f,ari=%.4f' %
                      (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred), ari(y, y_pred)))


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='liu_scrna1',
                        choices=['human_ESC', 'human_brain', 'time_course'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db
    if db == 'human_ESC':
        shape = 64
        x, y, dropout_matrix = get_datamat(db, shape, device)
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        padding = [0, 0, 0]
        epochs = 100
        weight_coef = 0.1
        weight_selfExp = 5
        weight_ot = 0.1
        weight = 1000
        batch_size = 32
        # post clustering parameters
        alpha = 0.06
        dim_subspace = 2
        ro = 1
    elif db == 'human_brain':
        shape = 64
        x, y, dropout_matrix = get_datamat(db, shape, device)
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        padding = [0, -1, 0]
        epochs = 100
        weight_coef = 0.1
        weight_selfExp = 5
        weight_ot = 0.01
        weight = 800
        batch_size = 32
        # post clustering parameters
        alpha = 0.02  # threshold of C
        dim_subspace = 5  # dimension of each subspace
        ro = 3
    elif db == 'time_course':
        shape = 64
        x, y, dropout_matrix = get_datamat(db, shape, device)
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        padding = [-1, 0, 0]
        epochs = 100
        weight_coef = 0.5
        weight_selfExp = 5
        weight_ot = 0.01
        weight = 1000  # coef weight
        batch_size = 32
        # post clustering parameters
        alpha = 0.02  # threshold of C 0.02
        dim_subspace = 5  # dimension of each subspace
        ro = 1  #
    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels, padding=padding)
    dscnet.to(device)
    pretrain_path = 'pretrained_weights_original/%s/%s.pkl' % (db, db)
    if not os.path.exists(pretrain_path):
        autoencoder = ConvAE(channels, kernels, padding)
        autoencoder.to(device)
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        pretrain_ae(autoencoder, optimizer_ae, x, epochs=10000, batch_size=batch_size, pretrain_path=pretrain_path)

    else:
        ae_state_dict = torch.load(pretrain_path)
        dscnet.ae.load_state_dict(ae_state_dict)
    coef = sio.loadmat('coef/%s/%s_%s.mat' % (db, db, weight))['coef']
    coef = normalize(coef, norm='max', axis=1)
    coef = torch.from_numpy(coef.astype(np.float32)).to(device)
    dscnet.reset_coef(coef)
    print("Pretrained ae weights are loaded successfully.")

    train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp, weight_ot=weight_ot,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)
