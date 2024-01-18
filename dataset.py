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

        if index in self.imgs:
            path, target = self.imgs[index]
        else:
            print("errors on __getitem__ from file_dataset")
            path, target = self.samples_dict[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

    def update_imgs_path_list(self, rm_indices):
        self.imgs = copy.deepcopy(self.samples_dict)
        for index in sorted(rm_indices, reverse=True):
            del self.imgs[index]

# CachedDataset
class CachedDataset(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            cache_length: int,
            extra_transform: Optional[Callable] = None,
            extra_target_transform: Optional[Callable] = None,
    ):
        self.cache_len = cache_length
        self.transform = extra_transform
        self.target_transform = extra_target_transform

        self.samples = dict()          # {‘index’: (memoryview(data), memoryview(target), reuse_factor, abs(loss))}
        self.evict_candidates_heap = []     # [ (-abs(loss), index) ] -> min heap
        self.idx_to_be_dismissed = set()    # { index }
        self.max_loss_candidates = -(10 ** 9)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        start = time.time()

        sample, target, _, _ = self.samples[index]
        # TODO: if normalized_transform are applied, denormalize -> to_pil_image(0.5*img+0.5)
        #sample = torchvision.transforms.functional.to_pil_image(torch.from_numpy(np.asarray(sample)))
        sample = torch.from_numpy(np.asarray(sample)).to(torch.float32)
        target = target
        #import matplotlib.pyplot as plt;    plt.imshow(np.asarray(sample)); plt.show(); exit()
        if index not in self.idx_to_be_dismissed:
            self.samples[index][2].value += 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

    def _cache(self, idx, sample, target, reuse_factor=0, loss=0):
        #self.samples[idx] = [ memoryview(sample.numpy()), memoryview(array.array('I', [target])), reuse_factor, abs(loss) ]
        self.samples[idx] = [ sample.to(torch.float16).numpy(), np.int64(target), mp.Value('i',reuse_factor), abs(loss) ]
        #self.samples[idx] = [ sample.numpy(), np.int64(target), Value('i', reuse_factor), Value('d', abs(loss)) ]
        #self.samples[idx] = shared_memory.ShareableList( [sample.numpy(), np.int64(target), reuse_factor, abs(loss)] )

    def cache_batch(self, possibly_batched_index, samples, targets, losses):
        """
        Args:
            idx (Tensor(dtype=int)): Index
            sample (Tensor):
            target (Tensor):
            loss (Tensor(dtype=float)): Loss tensor of samples.
        """
        possibly_batched_index = possibly_batched_index.to('cpu')
        samples = samples.to('cpu')
        targets = targets.to('cpu')
        losses = losses.to('cpu')

        losses_neg_abs = torch.mul(torch.abs(losses), -1)   # -abs(loss)
        loss_condi = torch.where(losses_neg_abs < self.max_loss_candidates, 0., 1.)

        idx_copy = copy.deepcopy(possibly_batched_index)
        idx_condi = idx_copy.apply_(lambda x: x not in self.samples).bool()

        condi = torch.mul(idx_condi, loss_condi)

        caching_idx = possibly_batched_index[condi == 1]
        caching_samples = samples[condi == 1]
        caching_targets = targets[condi == 1]
        caching_losses = losses[condi == 1]
        #print("cache_batch()", len(possibly_batched_index), len(caching_idx), mp.current_process())

        for (index, sample, target, loss) in zip(caching_idx, caching_samples, caching_targets, caching_losses):
            idx = index.item()

            if len(self.samples) < self.cache_len:
                '''
                `self.samples` has not been decided on the first epoch
                All data structures related to evict will be used in the same size as `self.samples`
                '''
                # insert
                heapq.heappush(self.evict_candidates_heap, (-abs(loss), idx))

            else:
                try:
                    popped_loss, popped_idx = heapq.heapreplace(self.evict_candidates_heap, (-abs(loss), idx))
                    _ = self.release_from_idx(popped_idx, has_to_delete_idx=True)
                except IndexError:
                    '''
                    If the current epoch is less than min_reuse_factor, the `self.evict_candidates_heap` might be empty.
                    '''
                    return

            self._cache(idx, sample, target, 0, loss)
            # self.idx_to_be_dismissed.add(idx)
            self.max_loss_candidates = self.evict_candidates_heap[0][0] if len(self.evict_candidates_heap) > 0 else -(10 ** 9)

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

    def make_evict_candidates(self, min_reuse_factor, evict_ratio):
        '''
        0. `update_reuse_factor()` for the remaining ones in `self.idx_to_be_dismissed`
        1. Making heap by scanning all the elements in `self.samples`: that `reuse_factor` exceeds min value
            -> heap has (reuse_factor, loss, index) as value
        2. Extract from heap using `heapq.nsmallest(num, q)`
        3. Remove the `reuse_factor` from the heap element tuple
        4. `update_imgs_path_list()`
        '''

        self.update_reuse_factor_for_remain_evicter()

        scan_evict_indices_heap = []    # [ (reuse_factor, abs(loss), index) ]
        for k, v in self.samples.items():
            if (v[2].value >= min_reuse_factor):
                heapq.heappush(scan_evict_indices_heap, (v[2].value, v[3], k))
        evict_candidates_heap = heapq.nlargest(int(self.cache_len * evict_ratio), scan_evict_indices_heap)

        self.evict_candidates_heap = [(-i[1], i[2]) for i in evict_candidates_heap]
        heapq.heapify(self.evict_candidates_heap)
        self.idx_to_be_dismissed = {i[2] for i in evict_candidates_heap}
        self.max_loss_candidates = self.evict_candidates_heap[0][0] if len(self.evict_candidates_heap) > 0 else -(10**9)

        self.update_imgs_path_list()

        del scan_evict_indices_heap, evict_candidates_heap
        gc.collect()    # invoke garbage collector manually

    def update_reuse_factor_for_remain_evicter(self):
        for idx in self.idx_to_be_dismissed:
            self.samples[idx][2].value += 1    # self.samples[idx] = (sample, target, reuse_factor, abs(loss))
        del self.idx_to_be_dismissed
