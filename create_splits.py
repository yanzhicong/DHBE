import os
import sys
from easydict import EasyDict
import pickle as pkl
import numpy as np

from dataloader import get_dataset



def generate_splits(output_dir, dataset_name, num_split, split_ind):

    train_ds, _ = get_dataset(EasyDict(dataset=dataset_name))
    targets = np.array(train_ds.targets, copy=True)
    classes = np.unique(targets)
    num_pre_class = int(num_split / len(classes))

    indices = []
    seed = split_ind
    for c in dataset_name:
        seed += ord(c)
    np.random.seed(seed)
    for cls in classes:
        indices += list(np.random.choice(np.where(targets==cls)[0], size=num_pre_class, replace=False))
    indices = np.array(indices)
    print("{} {} {} : {}".format(dataset_name, num_split, split_ind, len(indices)))
    pkl.dump(indices, open(os.path.join(output_dir, "splitind_{}_{}_{}.pkl".format(dataset_name, num_split, split_ind)), 'wb'))


if __name__ == "__main__":
    output_dir = "./dataset_split"
    os.makedirs(output_dir, exist_ok=True)

    for dataset in ["cifar10", "cifar100", "vggface2_subset", "mini-imagenet"]:
        for num_split in [ 250, 500, 1000, 2000, ]:
            for i in range(5):
                generate_splits(output_dir, dataset, num_split, i)
