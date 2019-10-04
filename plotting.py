import matplotlib.pyplot as plt

#training
loss_files = ['Model_1_A_sig_softmax_loss_train',
              'Model_1_B_sig_euc_loss_train',
              'Model_1_C_relu_softmaxent_loss_train',
              'Model_1_D_relu_euc_loss_train'
              ]

acc_files = ['Model_1_A_sig_softmax_acc_train',
              'Model_1_B_sig_euc_acc_train',
             'Model_1_C_relu_softmaxent_acc_train',
             'Model_1_D_relu_euc_acc_train'
              ]

#testing
acc_f_test = ['Model_1_A_sig_softmax_acc_test',
              'Model_1_B_sig_euc_acc_test',
             'Model_1_C_relu_softmaxent_acc_test',
             'Model_1_D_relu_euc_acc_test'
              ]


def plot_simple(a, name, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    plt.plot(a, label=name, linestyle='-')

    plt.legend()

    plt.show()


def plot(a, b, a_name, b_name, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    plt.plot(a, label=a_name, linestyle='-')
    plt.plot(b, label=b_name, linestyle='--')
    plt.legend()

    plt.show()

def plot4(a,an, b, bn, c, cn, d, dn, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    x = list(range(0, 100))

    plt.plot(x, a, label=an, linestyle=':')
    plt.plot(x, b, label=bn, linestyle='--')
    plt.plot(x, c, label=cn, linestyle='-.')
    plt.plot(x, d, label=dn, linestyle=':')
    plt.legend()

    plt.show()
def read_into_array(file):
    name = file
    arr = []
    file = open("results1/" + file, "r")
    for i in file.read().split(','):
        arr.append(float(i))
    return arr, name

test_acc = [0.1016, 0.19810, 0.69260, 0.81, 0.85190, 0.83610, 0.85660, 0.87910, 0.88160, 0.88350, 0.14990, ]
test_loss = [2.2989, 2.23933, 1.37694, 0.77668, 0.51643, 0.64750, 0.49993, 0.42045, 0.39737, 0.39438, 2.26844]

name = 'Model_1_C_relu_softmaxent_acc_train'
a2, a2n = read_into_array(acc_files[2])
# l1, l1n = read_into_array(loss_files[3])

plot(a2, test_acc, a2n, 'CNN Test accuracy', 'Best model 2 vs CNN', 'Accuracy', 'Epochs')
# plot(l1, test_loss, l1n, 'CNN Loss', 'Best model 2 vs CNN', 'Loss', 'Epoch')
plot_simple(test_loss, 'CNN Loss', 'Best model 2 vs CNN', 'Loss', 'Epochs')


# #loss SOFTMAX
# l0,l0n = read_into_array(loss_files[0])
# l1, l1n = read_into_array(loss_files[2])
# plot(l0,l1,l0n, l1n, 'Loss per epoch for Softmax-Entropy Model 2', "loss", "Epochs")
#
# #loss sumsqerr
# l0,l0n = read_into_array(loss_files[1])
# l1, l1n = read_into_array(loss_files[3])
# plot(l0,l1,l0n, l1n, 'Loss per epoch for Sum squared error Model 2', "loss", "Epochs")
#
# #acc train
# a0,a0n = read_into_array(acc_files[0])
# a1, a1n = read_into_array(acc_files[1])
# a2, a2n = read_into_array(acc_files[2])
# a3, a3n = read_into_array(acc_files[3])
#
# plot4(a0,a0n, a1, a1n, a2,a2n, a3, a3n,'Train accuracy per epoch for model 2', "Accuracy", "Epochs")
#
# #acc test
# a0,a0n = read_into_array(acc_f_test[0])
# a1, a1n = read_into_array(acc_f_test[1])
# a2, a2n = read_into_array(acc_f_test[2])
# a3, a3n = read_into_array(acc_f_test[3])
#
# plot4(a0,a0n, a1, a1n, a2,a2n, a3, a3n,'Test accuracy per epoch for model 2', "Accuracy", "Epochs")
