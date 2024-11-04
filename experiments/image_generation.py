import sys
sys.path.insert(0, "/Users/genna/Documents/ACSAI/Closed-Form DM/thesis") 

import argparse

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits

from algorithms import plain_smoothed, smoothed_from_T

datasets = {
    'cifar-10': ['truck', 'ship', 'horse', 'frog', 'dog', 'deer', 'cat', 'bird', 'automobile', 'airplane'],
    'mnist': [str(x) for x in range(10)],
}

def load_dataset(dataset, target):

    match dataset:
        case 'cifar-10':
            def unpickle(file, meta):
                import pickle
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                with open(meta, 'rb') as reader:
                    labels = pickle.load(reader, encoding='bytes')
                return dict, labels[b'label_names']
            dic, labels_names = unpickle('data/images/cifar-10/data_batch_1', 'data/images/cifar-10/batches.meta')
            imgs, labels_nums = dic[b'data'], dic[b'labels']

            imgs_labels_dict = {}
            for i in range(imgs.shape[1]):
                if labels_names[labels_nums[i]] in imgs_labels_dict:
                    imgs_labels_dict[labels_names[labels_nums[i]]].append(imgs[i])
                else:
                    imgs_labels_dict[labels_names[labels_nums[i]]] = []
                    imgs_labels_dict[labels_names[labels_nums[i]]].append(imgs[i])

            X = np.array(imgs_labels_dict[bytes(target, 'utf-8')])/255.0
    
        case 'mnist':
            mnist = load_digits()
            labels = mnist.target
            target = int(target)
            imgs = mnist.images[labels == int(target)]
            X = imgs.reshape(imgs.shape[0], 64)

    return X

def main(args):

    '''
    Experimenting generating images directly in the pixel space.
    '''

    N = 50
    X = load_dataset(args.dataset, args.target)[:N, :]
    fig, axs = plt.subplots(6, 6, figsize=(12, 12))
    fig.suptitle('Samples')
    
    sigmas_list = np.array(range(0, 50 + 1, 10)) / 10
    noises_list = np.array(range(5, 55 + 1, 10))

    # Rows and columns headers #
    pad = 5
    for ax, col in zip(axs[0], noises_list):
        ax.annotate(f'M = {col}', xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axs[:,0], sigmas_list):
        ax.annotate(f'σ = {row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    # fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    axs = axs.flatten()
    for (i, s) in enumerate(sigmas_list):
        for (j, M) in enumerate(noises_list):
            z = smoothed_from_T(
                X = X, 
                S = 1000,
                K = 1,
                sigma = s,
                M = M,
                start_T = 0.95
            )
            img = z[-1, -1, :]

            match args.dataset:
                case 'cifar-10': img.shape = (3, 32, 32); img = np.transpose(img, (1, 2, 0)); axs[i*len(noises_list) + j].imshow(img)
                case 'mnist': img.shape = (8, 8); axs[i*len(noises_list) + j].imshow(img, cmap='gray')


    # PLOT TRAINING SET #
    fig, axs = plt.subplots(6, 6, figsize=(12, 12))
    fig.suptitle('Training')
    axs = axs.flatten()
    for img, ax in zip(X, axs):
        match args.dataset:
            case 'mnist': img.shape = (8, 8)
            case 'cifar-10': img = img.reshape(3, 32, 32).transpose(1,2,0)
        ax.imshow(img, cmap='gray')

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='dataset'
    )
    
    for ds in datasets.keys():
        subparser = subparsers.add_parser(ds)
        subparser.add_argument(
            'target',
            choices = datasets[ds]
        )
    
    args = parser.parse_args()

    main(args)

