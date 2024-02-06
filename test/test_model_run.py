import torch, torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys, argparse
import test_model_utils as utils
import time

class ImageFolder_with_loading_time(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        start = time.time()

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

def plot_result(loss, acc, time, val_loss, val_acc, model_name, output_dir=None):
    # plot figure
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='row', sharey='col', layout="constrained")

    ax[0, 0].plot(range(len(loss)), loss, label='Train')
    ax[0, 1].plot(range(len(acc)), acc, label='Train')
    ax[1, 0].plot(np.cumsum(time), loss)
    ax[1, 1].plot(np.cumsum(time), acc)
    ax[0, 0].plot(range(len(val_loss)), val_loss, label='Validation')
    ax[0, 1].plot(range(len(val_acc)), val_acc, label='Validation')

    ax[0, 0].set_xlabel('Epochs');                ax[0, 1].set_xlabel('Epochs')
    ax[1, 0].set_xlabel('Elapsed Time (sec)');    ax[1, 1].set_xlabel('Elapsed Time (sec)')

    ax[0, 0].set_ylabel('Loss')
    ax[0, 1].set_ylabel('Accuracy')
    ax[1, 0].set_ylabel('Train Loss')
    ax[1, 1].set_ylabel('Train Accuracy')

    ax[0, 0].legend();
    ax[0, 1].legend();
    fig.suptitle(model_name)

    # plt.show()
    plt.savefig(output_dir + '/' + model_name + '.jpg', dpi=300)

def make_model(model, num_classes):
    # change the output layer to num_classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if __package__ is None:
        sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
        from dataset import ImageFolderWithCache, CachedDataset
        from dataloader import DataLoaderWithCache
        from transforms import basic_transform, autoaugment_transform, randaugment_transform

    else:
        from ..dataset import ImageFolderWithCache, CachedDataset
        from ..dataloader import DataLoaderWithCache
        from ..transforms import basic_transform, autoaugment_transform, randaugment_transform

    parser = argparse.ArgumentParser()

    parser.add_argument("--trainset", "-t", metavar='T', type=str,
                        nargs='?', default='/data/home/jmirrinae/ILSVRC2012/train', help='trainset path')
    parser.add_argument("--valset", "-v", metavar='V', type=str,
                        nargs='?', default=None, help='validationset path')
    parser.add_argument("--output", "-o", metavar='O', type=str,
                        nargs='?', default='./Imagenet2012', help='output directory path')
    parser.add_argument("--criteria", "-c", metavar='C', type=str, choices=['random', 'loss_sample'],
                        nargs='?', default='random', help='choose caching criteria. random or loss')
    parser.add_argument("--num_ops", "-n", metavar='N', type=int, nargs='?', default=1, help='number of augmentation block')
    parser.add_argument("--cache_ratio", "-r", metavar='R', type=float, nargs='?', default=0.05, help='')
    parser.add_argument("--evict_ratio", "-e", metavar='E', type=float, nargs='?', default=0.1, help='')
    parser.add_argument("--min_reuse_factor", "-m", metavar='M', type=int, nargs='?', default=1, help='number of reuse')
    parser.add_argument("--need_split", "-s", action="store_true", help='need to split dataset into train_set / test_set')
    args = parser.parse_args()

    a_t, a_t_b  = autoaugment_transform(basic_transform=basic_transform)
    r_t, r_t_b  = randaugment_transform(basic_transform=basic_transform, num_ops=args.num_ops)
    ra_t, _     = autoaugment_transform(basic_transform=basic_transform, p=args.cache_ratio)
    rr_t, _     = randaugment_transform(basic_transform=basic_transform, num_ops=args.num_ops, p=args.cache_ratio)

    loader_type = 'proposed'
    batch_size = 256

    # dataset
    if (loader_type == 'proposed'):
        train_image_folder = ImageFolderWithCache(root=args.trainset, transform=basic_transform)
        train_image_cache  = CachedDataset(cache_length=int(len(train_image_folder) * args.cache_ratio),
                                           evict_ratio=args.evict_ratio, min_reuse_factor=args.min_reuse_factor,
                                           extra_transform=a_t_b)
    elif (loader_type == 'default'):
        train_image_folder = ImageFolder_with_loading_time(root=args.trainset, transform=ra_t)
        train_image_cache  = None

    test_image_folder = ImageFolder_with_loading_time(root=args.valset, transform=basic_transform)

    # dataloader
    if (loader_type == 'proposed'):
        train_batch_num    = int( (len(train_image_folder) + batch_size - 1) // batch_size )
        train_dataloader   = DataLoaderWithCache(train_image_folder, batch_num=train_batch_num, shuffle=True,
                                                  num_workers=16, num_threads=2)
        train_cacheloader  = DataLoaderWithCache(train_image_cache, batch_num=train_batch_num)
        train_loader       = (train_dataloader, train_cacheloader)

    elif (loader_type == 'default'):
        train_loader       = torch.utils.data.DataLoader(train_image_folder, batch_size=batch_size, shuffle=True,
                                                          num_workers=16)
        train_batch_num    = len(train_loader)

    test_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=batch_size, shuffle=True, num_workers=16)

    # get deep learning model
    preset_model = torchvision.models.resnet18(weights=None)
    #preset_model = torchvision.models.alexnet(weights=None)
    #preset_model = torchvision.models.mobilenet_v2(weights=None)
    model = make_model(model=preset_model, num_classes=10)
    model_name = 'ResNet18'

    # make output directory
    if (not os.path.exists(args.output)):
        os.makedirs(args.output)

    # loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters())    #SGD(model.parameters(), lr=0.001)

    # train
    loss, acc, time, val_loss, val_acc, loading = utils._train_model(model, dataset=train_image_folder,
                                                                     cache_dataset=train_image_cache, batch_num=train_batch_num,
                                                                     train_loader=train_loader, test_loader=test_loader,
                                                                     epochs=100, device=device,
                                                                     criterion=criterion, optimizer=optimizer,
                                                                     model_name=model_name, output_dir=args.output,
                                                                     criteria=args.criteria)

    # save result
    result = {'train_loss':loss, 'train_accuracy':acc, 'elapsed_time':time, 'validation_loss':val_loss, 'validation_accuracy':val_acc, 'loading_time':loading}
    df = pd.DataFrame(result)
    df.to_csv(args.output+'/'+model_name+'.csv')
    
    # plot figure
    loss = df['train_loss'];    acc = df['train_accuracy'];    time = df['elapsed_time']
    val_loss = df['validation_loss'];    val_acc = df['validation_accuracy']
    plot_result(loss=loss, acc=acc, time=time, val_loss=val_loss, val_acc=val_acc, model_name=model_name, output_dir=args.output)

    # TODO: return objective function with `print()`
    print((0.4 * np.mean(val_acc)) - (0.6 * np.mean(loading)))