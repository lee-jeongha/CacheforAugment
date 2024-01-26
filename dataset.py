import torch, torchvision
import numpy as np
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import heapq, gc, time, copy
import multiprocessing as mp

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

        self.samples = dict()              # {‘index’: (data, target, reuse_factor, -abs(loss))}
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
        try:
            sample, target, reuse_factor, loss = self.samples[index]
        except KeyError:
            print(index, index in self.idx_to_be_dismissed)
        start = time.time()
        self.samples[index][2].value += 1

        # TODO: if normalized_transform are applied, denormalize -> to_pil_image(0.5*img+0.5)
        #sample = torchvision.transforms.functional.to_pil_image(torch.from_numpy(np.asarray(sample)))
        sample = torch.from_numpy(np.asarray(sample)).to(torch.float32)
        target = target
        #import matplotlib.pyplot as plt;    plt.imshow(np.asarray(sample)); plt.show(); exit()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

    def _cache(self, idx, sample, target, reuse_factor=0, loss=0):
        """
        Args:
            idx (int): Dataset Index
            sample (ndarray(dtype=float16)):
            target (ndarray(dtype=float32)):
            reuse_factor (int)
            loss (float): Negative Absolute Loss of samples
        """

        #self.samples[idx] = [ memoryview(sample), memoryview(array.array('I', [target])), reuse_factor, loss ]
        self.samples[idx] = [ sample, target, mp.Value('i',reuse_factor), loss ]
        #self.samples[idx] = [ sample, target, Value('i', reuse_factor), Value('d', loss) ]
        #self.samples[idx] = shared_memory.ShareableList( [sample, target, reuse_factor, loss] )

    def _temp_cache(self, caching_idx, caching_samples, caching_targets, caching_losses):
        """
        Args:
            caching_idx (List(int)): List of Dataset Index
            caching_samples (ndarray(dtype=float16)):
            caching_targets (ndarray(dtype=float32):
            caching_losses (List(float)): List of Negative Absolute Loss of samples
        """

        for (index, sample, target, loss) in zip(caching_idx, caching_samples, caching_targets, caching_losses):
            if (not len(self.samples)) and (len(self.temp_samples) < self.cache_length):
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

        if (len(self.idx_to_be_dismissed) <= 0) and (len(self.samples) >= self.cache_length):
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
        idx_condi = idx_copy.apply_(lambda x: x not in self.samples).bool()

        neg_abs_losses = torch.mul(torch.abs(losses), -1)   # -abs(loss)
        loss_condi = torch.where(neg_abs_losses < self.max_loss_candidates, 0., 1.)

        condi = torch.mul(idx_condi, loss_condi).bool()

        caching_idx = possibly_batched_index[condi == 1].tolist()
        caching_samples = samples[condi == 1].to(torch.float16).numpy()
        caching_targets = targets[condi == 1].numpy()
        caching_losses = neg_abs_losses[condi == 1].tolist()

        self._temp_cache(caching_idx, caching_samples, caching_targets, caching_losses)

        self.max_loss_candidates = self.temp_samples[0][0] if len(self.temp_samples) > 0 else -(10 ** 9)

    def release_from_idx(self, idx: int, has_to_delete_idx=False):
        sample, target, reuse_factor, _ = self.samples[idx]
        rf = reuse_factor.value

        #sample.release()    # memoryview
        #target.release()    # memoryview
        del sample
        del target
        del reuse_factor

        del self.samples[idx]

        if has_to_delete_idx:
            # Some elements that have been replaced in this epoch may not be in the `self.idx_to_be_dismissed`.
            # -> use `set().discard()` instead of `set().remove()`
            self.idx_to_be_dismissed.discard(idx)

        return rf

    def make_evict_candidates(self):
        """
        0. cache all elements in `self.temp_samples`
        1. Making heap by scanning all the elements in `self.samples`: that `reuse_factor` exceeds min value
            -> heap has (reuse_factor, loss, index) as value
        2. Extract from heap using `heapq.nsmallest(num, q)`
        3. Remove the `reuse_factor` from the heap element tuple
        4. `update_imgs_path_list()`
        """
        rm_indices, add_indices = self.cache_temp_samples()

        scan_evict_indices_heap = []    # [ (reuse_factor, -abs(loss), index) ]
        for k, v in self.samples.items():
            if (v[2].value >= self.min_reuse_factor):
                heapq.heappush(scan_evict_indices_heap, (v[2].value, v[3], k))
        evict_candidates_heap = heapq.nsmallest(self.evict_length, scan_evict_indices_heap)

        self.temp_samples = [(i[1], i[2]) for i in evict_candidates_heap]
        heapq.heapify(self.temp_samples)
        self.idx_to_be_dismissed = {i[2] for i in evict_candidates_heap}
        self.max_loss_candidates = evict_candidates_heap[0][1] if len(evict_candidates_heap) > 0 else -(10 ** 9)

        del scan_evict_indices_heap, evict_candidates_heap
        gc.collect()    # invoke garbage collector manually

        return rm_indices, add_indices

    def cache_temp_samples(self):
        current_indices = {ts[1] for ts in self.temp_samples}
        remain_indices = current_indices.intersection(self.idx_to_be_dismissed)
        rm_indices = self.idx_to_be_dismissed - remain_indices
        add_indices = remain_indices - self.idx_to_be_dismissed

        # for add in add_indices:
        for ts in self.temp_samples:
            try:
                self._cache(idx=ts[1], sample=ts[2], target=ts[3], reuse_factor=0, loss=ts[0])
            except IndexError:
                """
                Some of the elements in `self.idx_to_be_dismissed` that hasn't been evicted
                """
                assert ts[1] in self.idx_to_be_dismissed, "cache_temp_samples error."

        for rm in rm_indices:
            _ = self.release_from_idx(idx=rm, has_to_delete_idx=False)

        del self.temp_samples[:]

        return rm_indices, add_indices