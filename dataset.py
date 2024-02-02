import torch, torchvision
from tensordict import MemoryMappedTensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from typing import Any, Callable, Optional, Tuple, Sequence
from PIL import Image
import heapq, gc, time, copy
import multiprocessing as mp
from operator import itemgetter
import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

# ImageFolder
class ImageFolderWithCache(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        #self.imgs = self.samples
        self.samples_dict = { idx : sample for idx, sample in enumerate(self.samples) }
        self.imgs = copy.deepcopy(self.samples_dict)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        start = time.time()

        path, target = self.imgs[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

# Tensorclass
@tensorclass
class MemMapData:
    indices: torch.Tensor
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, cache_sample, batch_size, num_workers):
        data = cls(
            indices=MemoryMappedTensor.empty((len(cache_sample),), dtype=torch.int64),
            images=MemoryMappedTensor.empty(
                (
                    len(cache_sample),
                    *cache_sample[next(iter(cache_sample))][2].squeeze().shape,
                ),
                dtype=torch.float16,
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
        for loss, index, image, target in dl:
            _batch = image.shape[0]
            pbar.update(_batch)
            #print(data, type(data))
            #print(cls(images=image, targets=target, batch_size=[_batch]))
            data[i : i + _batch] = cls(
                indices=index, images=image, targets=target, batch_size=[_batch]
            )
            i += _batch

        return data

# CachedDataset
class CachedDataset(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            cache_length: int,
            evict_ratio: float = 0.1,
            min_reuse_factor: int = 1,
            extra_transform: Optional[Callable] = None,
            extra_target_transform: Optional[Callable] = None,
    ):
        self.cache_length = cache_length
        self.evict_length = int(self.cache_length * evict_ratio)
        self.min_reuse_factor = min_reuse_factor
        self.transform = extra_transform
        self.target_transform = extra_target_transform

        self.samples = TensorDict({'indices': torch.empty((0, 1), dtype=torch.int64),
                                   'images': torch.empty((0, 1), dtype=torch.float16),
                                   'targets': torch.empty((0, 1), dtype=torch.int64)}, batch_size=0)
        self.sample_info = dict()       # {‘index’: (position_index, reuse_factor, -abs(loss))}
        self.imgs = self.sample_info
        self.temp_samples = []             # [ (-abs(loss), index, data, target) ] -> min heap
        self.idx_to_be_dismissed = set()   # { index }
        self.max_loss_candidates = -(10 ** 9)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        start = time.time()

        try:
            pos_idx, _, _ = self.sample_info[index]
            sample = self.samples[pos_idx].images   #self.samples.get_at(key='images', index=pos_idx)
            target = self.samples[pos_idx].targets   #.get_at(key='targets', index=pos_idx)
        except KeyError:
            try:
                assert self.samples[pos_idx].indices.item() == index, "error on __getitem__"
            except AssertionError:
                print(self.samples[pos_idx].indices.item(), index)

        self.sample_info[index][1].value += 1

        # TODO: if normalized_transform are applied, denormalize -> to_pil_image(0.5*img+0.5)
        #sample = torchvision.transforms.functional.to_pil_image(sample)
        sample = sample.to(torch.float32)
        target = target
        #import matplotlib.pyplot as plt;    plt.imshow(np.asarray(sample)); plt.show(); exit()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

    def __getitems__(self, indices: Sequence[int]) -> Tuple[Any, Any]:
        """
        Args:
            indices (sequence[int]): a sequence of indices

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        start = time.time()

        try:
            sample_info_ = itemgetter(*indices)(self.sample_info)
            pos_indices = list(map(itemgetter(0), sample_info_))
            samples = self.samples.get_at(key='images', idx=pos_indices)   #.get_at(key='images', index=pos_indices)
            targets = self.samples.get_at(key='targets', idx=pos_indices)   #.get_at(key='targets', index=pos_indices)
        except KeyError:
            try:
                assert self.samples.get_at(key='indices', idx=pos_indices) == torch.tensor(indices), "error on __getitems__"
            except AssertionError:
                print(self.samples.get_at(key='indices', idx=pos_indices), indices)

        datas = []
        for i, (index, sample, target) in enumerate(zip(indices, samples, targets)):
            self.sample_info[index][1].value += 1

            # TODO: if normalized_transform are applied, denormalize -> to_pil_image(0.5*img+0.5)
            #sample = torchvision.transforms.functional.to_pil_image(sample)
            sample = samples[i].to(torch.float32)
            target = targets[i]
            #import matplotlib.pyplot as plt;    plt.imshow(np.asarray(sample)); plt.show(); exit()

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            end = time.time()
            datas.append((index, sample, target, end-start))

        #datas = []
        #for index, sample, target in zip(indices, samples, targets):
        #    datas.append((index, sample, target, (end-start) / len(indices)))

        #return indices, samples, targets, [(end-start) / len(indices)]*len(indices)
        return datas

    def _temp_cache(self, caching_idx, caching_samples, caching_targets, caching_losses):
        """
        Args:
            caching_idx (List(int)): List of Dataset Index
            caching_samples (ndarray(dtype=float16)):
            caching_targets (ndarray(dtype=float32):
            caching_losses (List(float)): List of Negative Absolute Loss of samples
        """

        for (index, sample, target, loss) in zip(caching_idx, caching_samples, caching_targets, caching_losses):
            if (not len(self.sample_info)) and (len(self.temp_samples) < self.cache_length):
                """
                Case 2. On the first epoch
                `self.samples` has not been decided
                All data structures related to evict will be used in the same size as `self.samples`
                """
                # insert
                heapq.heappush(self.temp_samples, (loss, index, sample, target))

            else:
                try:
                    """evicted == (e_loss, e_idx, e_sample, e_target, e_reuse_factor)"""
                    evicted = heapq.heappushpop(self.temp_samples, (loss, index, sample, target))
                    del evicted
                except IndexError:
                    print("error on cache_batch: temp_sample is empty")
                    return

        '''cache_data = list(zip(caching_losses, caching_idx, caching_samples, caching_targets))
        heapq.heapify(cache_data)

        if (not len(self.samples)) and (len(self.temp_samples) < self.cache_length):
            """
            Case 2. On the first epoch
            `self.samples` has not been decided
            All data structures related to evict will be used in the same size as `self.samples`
            """
            n = self.cache_length - len(self.temp_samples)

            # insert
            self.temp_samples = list(heapq.merge(self.temp_samples, cache_data[:n]))
            # replace
            for cd in cache_data[n:]:
                evicted = heapq.heappushpop(self.temp_samples, cd)
                del evicted

        else:
            try:
                # replace
                for cd in cache_data:
                    """evicted == (e_loss, e_idx, e_sample, e_target, e_reuse_factor)"""
                    evicted = heapq.heappushpop(self.temp_samples, cd)
                    del evicted
            except IndexError:
                print("error on cache_batch: temp_sample is empty")
                return
        #print("cache:", time.time() - start); exit()
        del cache_data'''

    def cache_batch(self, possibly_batched_index, samples, targets, losses):
        """
        Args:
            idx (Tensor(dtype=int)): Index
            sample (Tensor):
            target (Tensor):
            loss (Tensor(dtype=float)): Loss tensor of samples.
        """

        if (len(self.idx_to_be_dismissed) <= 0) and (len(self.sample_info) >= self.cache_length):
            """
            Case 1. If the current epoch is less than `self.min_reuse_factor`
            `self.idx_to_be_dismissed` might be empty.
            """
            return

        possibly_batched_index = possibly_batched_index.to('cpu')
        samples = samples.to('cpu').detach()
        targets = targets.to('cpu')
        losses = losses.to('cpu').detach()

        idx_copy = copy.deepcopy(possibly_batched_index)
        idx_condi = idx_copy.apply_(lambda x: x not in self.sample_info).bool()

        neg_abs_losses = torch.mul(torch.abs(losses), -1)   # -abs(loss)
        loss_condi = torch.where(neg_abs_losses < self.max_loss_candidates, 0., 1.)

        condi = torch.mul(idx_condi, loss_condi).bool()

        caching_idx = possibly_batched_index[condi == 1].tolist()
        caching_samples = samples[condi == 1].to(torch.float16)#.numpy()
        caching_targets = targets[condi == 1].tolist()#numpy()
        caching_losses = neg_abs_losses[condi == 1].tolist()

        self._temp_cache(caching_idx, caching_samples, caching_targets, caching_losses)

        self.max_loss_candidates = self.temp_samples[0][0] if len(self.temp_samples) > 0 else -(10 ** 9)

    def make_evict_candidates(self):
        """
        0. cache all elements in `self.temp_samples`
        1. Making heap by scanning all the elements in `self.sample_info`: that `reuse_factor` exceeds min value
            -> heap has (reuse_factor, loss, index) as value
        2. Extract from heap using `heapq.nsmallest(num, q)`
        3. Remove the `reuse_factor` from the heap element tuple
        4. `update_imgs_path_list()`
        """
        self.cache_temp_samples()

        scan_evict_indices_heap = []    # [ (reuse_factor, -abs(loss), index) ]
        for k, v in self.sample_info.items():    # {‘index’: (position_index, reuse_factor, -abs(loss))}
            if (v[1].value >= self.min_reuse_factor):
                heapq.heappush(scan_evict_indices_heap, (v[1].value, v[2], k))
        evict_candidates_heap = heapq.nsmallest(self.evict_length, scan_evict_indices_heap)

        self.temp_samples = [(i[1], i[2]) for i in evict_candidates_heap]
        heapq.heapify(self.temp_samples)
        self.idx_to_be_dismissed = {i[2] for i in evict_candidates_heap}
        self.max_loss_candidates = evict_candidates_heap[0][1] if len(evict_candidates_heap) > 0 else -(10 ** 9)

        del scan_evict_indices_heap, evict_candidates_heap
        gc.collect()    # invoke garbage collector manually

        return

    def cache_temp_samples(self):
        temp_samples_dict = {ts[1]: tuple(ts) for ts in self.temp_samples} # {index: (loss, index, sample, target)}

        if len(self.samples) == 0 and len(self.temp_samples) == self.cache_length:
            self.samples = MemMapData.from_dataset(temp_samples_dict, batch_size=64, num_workers=16)

            for pos_idx, samp_idx in enumerate(self.samples.indices):
                index = samp_idx.item()
                self.sample_info[index] = (pos_idx, mp.Value('i', 0), temp_samples_dict[index][0])
            return

        elif len(self.samples) == 0:
            return

        current_indices = {ts[1] for ts in self.temp_samples}
        remain_indices = current_indices.intersection(self.idx_to_be_dismissed)
        rm_indices = list(self.idx_to_be_dismissed - remain_indices)
        add_indices = list(current_indices - remain_indices)

        assert len(rm_indices) == len(add_indices), "rm_indices and add_indices must be same length."

        if len(rm_indices) <= 0:
            return

        rm_ = itemgetter(*rm_indices)(self.sample_info) # {index: (position_index, reuse_factor, -abs(loss))}
        add_ = itemgetter(*add_indices)(temp_samples_dict) # {index: (loss, index, sample, target)}

        rm_position = [rm[0] for rm in rm_]
        add_images = torch.stack(list(map(itemgetter(2), add_)))
        add_targets = torch.tensor(list(map(itemgetter(3), add_)))

        self.samples.set_at_(key='indices', value=add_indices, idx=rm_position) #, index=rm_position)
        self.samples.set_at_(key='images', value=add_images, idx=rm_position) #, index=rm_position)
        self.samples.set_at_(key='targets', value=add_targets, idx=rm_position) #, index=rm_position)

        for add, rm, pos, elements in zip(add_indices, rm_indices, rm_position, add_):
            rm_elements = self.sample_info.pop(rm)
            #assert rm_elements[0] == pos
            #assert add == elements[1]
            self.sample_info[add] = (pos, mp.Value('i', 0), elements[0])

        del self.temp_samples[:], temp_samples_dict

        return