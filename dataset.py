import torch, torchvision
from torchvision.datasets.folder import *
from tensordict import MemoryMappedTensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Callable, Optional, Tuple, Sequence
import os, heapq, gc, time, copy
from heapq import _heapreplace_max, _heapify_max, _siftup_max, _siftdown_max
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
import tqdm
from transforms import transform_for_cache

# ImageFolder
class ImageFolderWithCache(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            cache_ratio: float = None,
            reuse_factor: int = 1,
            extra_transform: Optional[Callable] = None,
            extra_target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.items_dict = { idx : sample for idx, sample in enumerate(self.samples) }
        self.storage_items_dict = copy.deepcopy(self.items_dict)

        self.cache_length = int(len(self) * cache_ratio)
        self.reuse_factor = reuse_factor
        self.reuse_count = 0
        self.extra_transform = extra_transform
        self.extra_target_transform = extra_target_transform

        self.cache = TensorDict({'indices': torch.empty((0, 1), dtype=torch.int64),
                                 'samples': torch.empty((0, 1), dtype=torch.uint8),
                                 'targets': torch.empty((0, 1), dtype=torch.int64)}, batch_size=0)
        self.cache_info = dict()          # {'index': (position_index, -abs(loss))}
        self.temp_cache = []             # [ (-abs(loss), index) ] or [ (-abs(loss), index, sample, target) ] -> min heap
        self.idx_to_be_dismissed = set()  # { index }
        self.loss_criterion = -(10 ** 9)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (index, sample, target, load_time) where target is class_index of the target class.
        """
        start = time.time()

        is_in_cache = True if index in self.cache_info else False

        if not is_in_cache:
            index, sample, target = self.getitem_from_storage(index)
        else:
            index, sample, target = self.getitem_from_memcache(index)

        return index, sample, target, (time.time() - start)

    def getitem_from_storage(self, index: int):
        path, target = self.storage_items_dict[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, target

    def getitem_from_cache(self, index: int):
        pos_idx, _ = self.cache_info[index]
        try:
            sample = self.cache[pos_idx].samples    #self.cache.get_at(key='samples', index=pos_idx)
            target = self.cache[pos_idx].targets   #self.cache.get_at(key='targets', index=pos_idx)
        except KeyError:
            try:
                assert self.cache[pos_idx].indices.item() == index
            except AssertionError:
                print("error on __getitem__", self.cache[pos_idx].indices.item(), index)

        if self.extra_transform is not None:
            sample = self.extra_transform(sample)
        if self.target_transform is not None:
            target = self.extra_target_transform(target)

        return index, sample, target

    def __getitems__(self, indices: Sequence[int]) -> Tuple[Any, Any]:
        """
        Args:
            indices (sequence[int]): a sequence of indices

        Returns:
            tuple: (index, sample, target, load_time) where target is class_index of the target class.
        """
        start = time.time()

        if isinstance(indices, tuple):
            storage_indices = indices[0]
            cache_indices = indices[1]
        else:
            # filter
            idx_copy = copy.deepcopy(indices)
            is_in_cache = idx_copy.apply_(lambda x: x in self.cache_info).bool()
            storage_indices = indices[is_in_cache == False]
            cache_indices = indices[is_in_cache == True]

        #data = []
        #with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        #    exes = [executor.submit(self.__getitem__, idx) for idx in indices]
        #    data += [tuple(exe.result()) for exe in exes]

        from_storage = self.getitems_from_storage(storage_indices)
        from_cache = self.getitems_from_cache(cache_indices)
        data = from_storage + from_cache

        end = time.time()
        times = [(end-start) / len(data)] * len(data)

        data = list((d[0], d[1], d[2], t) for (d, t) in zip(data, times))

        return data

    def getitems_from_storage(self, indices: Sequence[int]):
        if indices is not None:
            start = time.time()

            items_ = itemgetter(*indices)(self.storage_items_dict) # list of (path, target)
            samples_path = list(map(itemgetter(0), items_))

            #samples = []
            #with ThreadPoolExecutor(max_workers=len(samples_path)) as executor:
            #    exes = [executor.submit(self.loader, path) for path in samples_path]
            #    samples += [exe.result() for exe in exes]
            samples = [self.loader(path) for path in samples_path]
            targets = list(map(itemgetter(1), items_))

            if self.transform is not None:
                samples = [self.transform(s) for s in samples]

            if self.target_transform is not None:
                targets = [self.target_transform(t) for t in targets]

            return list(zip(indices, samples, targets))#, [(time.time() - start) / len(indices)] * len(indices)))
        else:
            return list()

    def getitems_from_cache(self, indices: Sequence[int]):
        if indices is not None:
            start = time.time()

            cache_info_ = itemgetter(*indices)(self.cache_info)
            pos_indices = list(map(itemgetter(0), cache_info_))

            try:
                samples = self.cache.get_at(key='samples', idx=pos_indices)  #, index=pos_indices)
                targets = self.cache.get_at(key='targets', idx=pos_indices) #, index=pos_indices)
            except KeyError:
                try:
                    assert self.cache.get_at(key='indices', idx=pos_indices) == torch.tensor(indices)
                except AssertionError:
                    print("error on __getitems__:", self.cache.get_at(key='indices', idx=pos_indices), indices)

            if self.extra_transform is not None:
                samples = [self.extra_transform(s) for s in samples]

            if self.extra_target_transform is not None:
                targets = [self.extra_target_transform(t) for t in targets]

            return list(zip(indices, samples, targets))
        else:
            return list()

    def filter_batch_for_caching(self, possibly_batched_index, samples, targets, losses):
        """
        Returns:
            caching_idx (List(int)): List of Dataset Index
            caching_samples (torch.tensor(dtype=uint8)):
            caching_targets (torch.tensor(dtype=float32):
            caching_losses (List(float)): List of Negative Absolute Loss of samples
        """
        idx_copy = copy.deepcopy(possibly_batched_index)
        idx_condi = idx_copy.apply_(lambda x: x not in self.cache_info).bool()

        neg_abs_losses = torch.mul(torch.abs(losses), -1)   # -abs(loss)
        loss_condi = torch.where(neg_abs_losses < self.loss_criterion, 0., 1.)

        condi = torch.mul(idx_condi, loss_condi)

        caching_idx = possibly_batched_index[condi == 1].tolist()
        caching_samples = transform_for_cache(samples[condi == 1])    # samples[condi == 1].to(torch.float16)
        caching_targets = targets[condi == 1]                         # targets[condi == 1].tolist()
        caching_losses = neg_abs_losses[condi == 1].tolist()

        return caching_idx, caching_samples, caching_targets, caching_losses

    def replace_cache_item(self, rm_indices, rm_position, add_losses, add_indices, add_samples, add_targets):
        self.cache.set_at_(key='indices', value=add_indices, idx=rm_position) #, index=rm_position)
        self.cache.set_at_(key='samples', value=add_samples, idx=rm_position)   #, index=rm_position)
        self.cache.set_at_(key='targets', value=add_targets, idx=rm_position) #, index=rm_position)

        for add, rm, pos, loss in zip(add_indices, rm_indices, rm_position, add_losses):
            rm_elements = self.cache_info.pop(rm)
            self.cache_info[add] = (pos, loss)

        del rm_indices, rm_position, add_losses, add_indices, add_samples, add_targets

    def cache_batch(self, possibly_batched_index, samples, targets, losses):
        """
        Args:
            idx (Tensor(dtype=int)): Index
            sample (Tensor):
            target (Tensor):
            loss (Tensor(dtype=float)): Loss tensor of samples.
        """

        possibly_batched_index = possibly_batched_index.to('cpu')

        # If it is in self.idx_to_be_dismissed -> True,  else -> False
        dismissed_idx = list(filter(lambda x: x in self.idx_to_be_dismissed, possibly_batched_index.tolist()))
        dismissed_info = [self.cache_info[d_idx] for d_idx in dismissed_idx]
        for (idx, info) in zip(dismissed_idx, dismissed_info):
            heapq.heappush(self.temp_cache, (info[1], idx))

        #if (len(self.idx_to_be_dismissed) <= 0) and (len(self.cache_info) >= self.cache_length):
        if (self.reuse_count < self.reuse_factor) and (len(self.cache_info) >= self.cache_length):
            """
            Case 1. If the current epoch is less than `self.reuse_factor`
            `self.idx_to_be_dismissed` might be empty.
            """
            return

        #possibly_batched_index = possibly_batched_index.to('cpu')
        samples = samples.to('cpu').detach()
        targets = targets.to('cpu')
        losses = losses.to('cpu').detach()

        (caching_idx, caching_samples,
         caching_targets, caching_losses) = self.filter_batch_for_caching(possibly_batched_index, samples, targets, losses)

        cache_data = list(zip(caching_losses, caching_idx, caching_samples, caching_targets))
        cache_data = sorted(cache_data, key=itemgetter(0))

        if (not len(self.cache_info)) and (len(self.temp_cache) < self.cache_length):
            """
            Case 2. On the first epoch
            `self.cache` has not been decided
            All data structures related to evict will be used in the same size as `self.cache`
            """
            n = self.cache_length - len(self.temp_cache)

            # insert
            self.temp_cache = list(heapq.merge(self.temp_cache, cache_data[:n]))
            # replace
            for cd in cache_data[n:]:
                evicted = heapq.heappushpop(self.temp_cache, cd)
                del evicted

        else:
            """
            Case 3. Replace
            (1) updating = [-2, -2, -1, -1, 0],   current = [-4, -3, -2, 0, 0]
                -> mask = [1, 1, 1, 0, 0]
                -> replace_num = 3
                -> current[3:] + updating[-3:] = [0, 0] + [-1, -1, 0]
            (2) updating = [-5, -4, -3, -2, -1],  current = [-4, -4, -4, -3, -3]
                -> mask = [0, 0, 1, 1, 1]
                -> replace_num = 3
                -> current[3:] + updating[-3:] = [-3, -3] + [-3, -2, -1]
            (3) updating = [-7, -4, -3, -1, 0],   current = [-15, -9, -7, -3, -1]
                -> mask = [1, 1, 1, 1, 1]
                -> replace_num = 5
                -> updating[-5:] = [-7, -4, -3, -1, 0]
                * optimal = [-3, -3, -1, -1, 0]
            """
            evict_candidates = heapq_nsmallest_with_index(len(caching_idx), self.temp_cache)
            cache_data = cache_data[:len(evict_candidates)]
            # self.temp_cache  = [(loss, index, sample, target)] for data in current epoch
            #                  = [(loss, index)]                 for self.idx_to_be_dismissed
            # evict_candidates = [(elem_in_`self.temp_cache`, idx_from_`self.temp_cache`)]

            mask = [1 if c[0] >= e[0][0] else 0 for (c, e) in zip(cache_data, evict_candidates)]
            replace_num = mask.count(1)

            if (not len(self.cache)) or (not replace_num):
                pass
            else:
                # replace to `self.cache` and `self.cache_info`
                rm_indices = [e[0][1] for e in evict_candidates[:replace_num]]
                rm_position = [self.cache_info[rm][0] for rm in rm_indices]

                add_losses = list(map(itemgetter(0), cache_data[-replace_num:]))
                add_indices = list(map(itemgetter(1), cache_data[-replace_num:]))
                add_samples = torch.stack(list(map(itemgetter(2), cache_data[-replace_num:])))
                add_targets = torch.tensor(list(map(itemgetter(3), cache_data[-replace_num:])))

                self.replace_cache_item(rm_indices, rm_position, add_losses, add_indices, add_samples, add_targets)

                # save only losses and indices in `self.temp_cache`
                cache_data = list(map(itemgetter(0,1), cache_data))

            # replace to `self.temp_cache`
            for (c, e) in zip(cache_data[-replace_num:], evict_candidates[:replace_num]):
                pos = e[1]
                self.temp_cache[pos] = c
            heapq.heapify(self.temp_cache)

            del evict_candidates

        self.loss_criterion = self.temp_cache[0][0] if len(self.temp_cache) > 0 else -(10 ** 9)

    def make_evict_candidates(self):
        """
        1. cache all elements in `self.temp_cache`
        2. update `self.storage_items_dict` and `self.idx_to_be_dismissed`
        """
        if (not len(self.cache)) and len(self.temp_cache):
            self.cache_from_temp_cached()
            self.update_storage_items()

        elif self.reuse_count >= self.reuse_factor:
            self.reuse_count = 0
            self.update_storage_items()

        else:
            self.reuse_count += 1

        del self.temp_cache[:]
        gc.collect()    # invoke garbage collector manually

        return

    def update_storage_items(self):
        self.idx_to_be_dismissed = set(self.cache_info.keys())

        self.storage_items_dict = copy.deepcopy(self.items_dict)
        for index in sorted(self.cache_info.keys(), reverse=True):
            del self.storage_items_dict[index]

    def cache_from_temp_cached(self):
        temp_cached_dict = {ts[1]: tuple(ts) for ts in self.temp_cache}  # {index: (loss, index, sample, target)}

        self.cache = MemMapData.from_dataset(temp_cached_dict, batch_size=64, num_workers=16)

        for pos_idx, samp_idx in enumerate(self.cache.indices):
            index = samp_idx.item()
            self.cache_info[index] = (pos_idx, temp_cached_dict[index][0])
        return

# Find n smallest elements and thier location from heapq
def heapq_nsmallest_with_index(n, iterable, key=None):
    """Find the n smallest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key)[:n]

    Source: https://github.com/python/cpython/blob/3.12/Lib/heapq.py
    """

    # Short-cut for n==1 is to use min()
    if n == 1:
        it = iter(iterable)
        sentinel = object()
        #result = min(it, default=sentinel, key=key)
        if key is None:
            it = [(None, elem, idx) for idx, elem in enumerate(it)]
        else:
            it = [(key(elem), elem, idx) for idx, elem in enumerate(it)]
        result = min(it, default=sentinel, key=key)
        return [] if result is sentinel else [result[1:]]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            it = [(elem, idx) for idx, elem in enumerate(iter(iterable))]
            return sorted(it, key=key)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = iter(iterable)
        # put the range(n) first so that zip() doesn't
        # consume one too many elements from the iterator
        result = [(elem, i, i) for i, elem in zip(range(n), it)]
        if not result:
            return result
        _heapify_max(result)
        top = result[0][0]
        order = n
        _heapreplace = _heapreplace_max
        for idx, elem in enumerate(it):
            if elem < top:
                _heapreplace(result, (elem, order, n+idx))
                top, _order, _idx = result[0]
                order += 1
        result.sort()
        return [(elem, idx) for (elem, order, idx) in result]

    # General case, slowest method
    it = iter(iterable)
    result = [(key(elem), i, elem, i) for i, elem in zip(range(n), it)]
    if not result:
        return result
    _heapify_max(result)
    top = result[0][0]
    order = n
    _heapreplace = _heapreplace_max
    for idx, elem in enumerate(it):
        k = key(elem)
        if k < top:
            _heapreplace(result, (k, order, elem, n+idx))
            top, _order, _elem, _idx = result[0]
            order += 1
    result.sort()
    return [(elem, idx) for (k, order, elem, idx) in result]

# Tensorclass
@tensorclass
class MemMapData:
    indices: torch.Tensor
    samples: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, cache_sample, batch_size, num_workers=4):
        NUM_WORKERS = int(os.environ.get("NUM_WORKERS", str(num_workers)))

        data = cls(
            indices=MemoryMappedTensor.empty((len(cache_sample),), dtype=torch.int64),
            samples=MemoryMappedTensor.empty(
                (
                    len(cache_sample),
                    *cache_sample[next(iter(cache_sample))][2].squeeze().shape,
                ),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor.empty((len(cache_sample),), dtype=torch.int64),
            batch_size=[len(cache_sample)],
        )
        # locks the tensorclass and ensures that is_memmap will return True.
        data.memmap_()

        dl = DataLoader(cache_sample, batch_size=batch_size, num_workers=num_workers,
                        sampler=SubsetRandomSampler(list(cache_sample.keys())))
        i = 0
        pbar = tqdm.tqdm(total=len(cache_sample))
        for loss, index, sample, target in dl:
            _batch = sample.shape[0]
            pbar.update(_batch)
            #print(data, type(data))
            #print(cls(samples=sample, targets=target, batch_size=[_batch]))
            data[i : i + _batch] = cls(
                indices=index, samples=sample, targets=target, batch_size=[_batch]
            )
            i += _batch

        return data