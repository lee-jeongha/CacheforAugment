import torch, torchvision
import multiprocessing as py_multiprocessing
import os, sys, time

torch.multiprocessing.set_sharing_strategy('file_system')   # OSError: [Errno 24] Too many open files

def run_proposed(root, cache_ratio, transform, transform_block, min_reuse_factor, evict_ratio, criteria='random', epochs=5):
    mvif = ImageFolderWithCache(root=root, transform=transform)
    mvcd = CachedDataset(cache_length=int(len(mvif) * cache_ratio), evict_ratio=evict_ratio,
                         min_reuse_factor=min_reuse_factor, extra_transform=transform_block)
    dl = DataLoaderWithCache(mvif, mvcd, batch_size=16, shuffle=True, num_workers=4)#, num_threads=2)

    for epoch in range(epochs):
        start = time.time()
        #print(py_multiprocessing.current_process().__dict__['_name'], py_multiprocessing.active_children())
        for (idx, data, target, _) in dl:
            d, t = (data, target)

            if criteria == 'random':
                mvcd.cache_batch(idx, data, target, torch.rand(len(idx)))
            else:
                raise Exception("criteria has to be 'random'")
                #mvcd.cache_batch(idx, data, target, loss)

        end = time.time()
        print(end - start)

def run_default(root, transform, epochs=5):
    mvif = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dl = torch.utils.data.DataLoader(mvif, batch_size=16, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        start = time.time()

        for idx, (data, target) in enumerate(dl):
            d, t = (data, target)

        end = time.time()
        print(end - start)

if __name__ == '__main__':

    if __package__ is None:
        print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
        sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
        from dataset import ImageFolderWithCache, CachedDataset
        from dataloader import DataLoaderWithCache
        from transforms import basic_transform, autoaugment_transform, randaugment_transform

    else:
        from ..dataset import ImageFolderWithCache, CachedDataset
        from ..dataloader import DataLoaderWithCache
        from ..transforms import basic_transform, autoaugment_transform, randaugment_transform

    cache_ratio = 0.2
    epochs = 5
    root = '~/Desktop/CIFAR10/train'

    a_t, a_t_b = autoaugment_transform(basic_transform=basic_transform)
    r_t, r_t_b = randaugment_transform(basic_transform=basic_transform, num_ops=2)
    ra_t, ra_t_b = autoaugment_transform(basic_transform=basic_transform, p=cache_ratio)
    rr_t, _ = randaugment_transform(basic_transform=basic_transform, num_ops=2, p=cache_ratio)

    run_default(root=root, transform=ra_t, epochs=epochs)
    run_proposed(root=root, cache_ratio=cache_ratio, transform=basic_transform, transform_block=a_t_b,
                 min_reuse_factor=2, evict_ratio=0.2, criteria='random', epochs=epochs)