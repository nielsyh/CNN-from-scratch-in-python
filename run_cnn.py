from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
# model.add(Reshape('flatten', (-1, 588)))
model.add(Linear('fc3', 196, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 0.0,
    'momentum': 0.001,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
}

def save_file(name, array):
        with open(name, 'w') as file:
            for i in array:
                file.write('%s,' % i)
            file.close()

acc_train = []
acc_test = []
loss_train = []
loss_test = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    tmp_loss, tmp_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    acc_train.append(tmp_acc)
    loss_train.append(tmp_loss)

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    tmp_loss, tmp_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
    acc_test.append(tmp_acc)
    loss_test.append(tmp_loss)

    name = 'lr0.01mm0.01'
    save_file(name + '_loss_train', loss_train)
    save_file(name + '_acc_train', acc_train)
    save_file(name + '_loss_test', loss_test)
    save_file(name + '_acc_test', acc_test)
