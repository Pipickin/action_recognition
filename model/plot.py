from matplotlib import pyplot as plt


def save_loss(train_loss, test_loss, save_path):
    fig, ax = plt.subplots()
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.savefig(save_path)


def save_accuracy(train_acc, test_acc, save_path):
    fig, ax = plt.subplots()
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.savefig(save_path)


def save_all_plots(train_loss, test_loss, train_acc, test_acc, loss_path, acc_path):
    save_loss(train_loss, test_loss, loss_path)
    save_accuracy(train_acc, test_acc, acc_path)

