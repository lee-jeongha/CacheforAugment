import torch, torchvision
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
import heapq, array, gc, time
from multiprocessing import shared_memory, Value

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
        cache_ratio: Optional[float] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transform_block: Optional[Callable] = None,
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
        self.imgs = self.samples
        self.transform_block = transform_block
        self.cache_len = int(len(self) * cache_ratio)
        self.cache_sample = dict()          # {‘index’: (memoryview(data), memoryview(target), reuse_factor, abs(loss))}
        self.evict_candidates_heap = []     # [ (-abs(loss), index) ] -> min heap
        self.idx_to_be_dismissed = set()    # { index }
        self.max_loss_candidates = -(10 ** 9)
        #self.cache_func = self.cache_from_idx_first_epoch

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        start = time.time()

        if index in self.cache_sample:
            sample, target, _, _ = self.cache_sample[index]
            # TODO: if normalized_transform are applied, denormalize -> to_pil_image(0.5*img+0.5)
            #sample = torchvision.transforms.functional.to_pil_image(torch.from_numpy(np.asarray(sample)))
            sample = torch.from_numpy(np.asarray(sample)).to(torch.float32)
            target = target
            #import matplotlib.pyplot as plt;    plt.imshow(np.asarray(sample)); plt.show(); exit()
            if index not in self.idx_to_be_dismissed:
                self.cache_sample[index][2] += 1
            if self.transform_block is not None:
                sample = self.transform_block(sample)
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        end = time.time()

        return index, sample, target, (end-start)

    def _cache(self, idx, sample, target, reuse_factor=0, loss=0):
        #self.cache_sample[idx] = [ memoryview(sample.numpy()), memoryview(array.array('I', [target])), reuse_factor, abs(loss) ]
        self.cache_sample[idx] = [ sample.to(torch.float16).numpy(), np.int64(target), reuse_factor, abs(loss) ]
        #self.cache_sample[idx] = [ sample.numpy(), np.int64(target), Value('i', reuse_factor), Value('d', abs(loss)) ]
        #self.cache_sample[idx] = shared_memory.ShareableList( [sample.numpy(), np.int64(target), reuse_factor, abs(loss)] )

    def cache_from_idx(self, idx: int, sample, target, loss):
        if idx in self.cache_sample:
            pass

        if -abs(loss) < self.max_loss_candidates:
            return

        # TODO: if already in `idx_to_be_dismissed` -> update?

        elif idx not in self.cache_sample:
            # replace
            try:
                heapq.heapify(self.evict_candidates_heap)
                popped_loss, popped_idx = heapq.heapreplace(self.evict_candidates_heap, (-abs(loss), idx))
            except IndexError:  # If the current epoch is less than min_reuse_factor, the `self.evict_candiates_heap` might be empty.
                return
            #self.idx_to_be_dismissed.add(idx)
            self.max_loss_candidates = self.evict_candidates_heap[0][0] if len(self.evict_candidates_heap) > 0 else -(10**9)

            _ = self.release_from_idx(popped_idx, has_to_delete_idx=True)

            self._cache(idx, sample, target, 0, loss)

    def cache_from_idx_first_epoch(self, idx: int, sample, target, loss):
        '''
        `self.cache_sample` has not been decided on the first epoch
        All data structures related to evict will be used in the same size as `self.cache_sample`
        '''
        if len(self.cache_sample) < self.cache_len:
            # insert
            self._cache(idx, sample, target, 0, loss)

            #heapq.heappush(self.evict_candidates_heap, (-abs(loss), idx))
            self.evict_candidates_heap.append((-abs(loss), idx))
            heapq.heapify(self.evict_candidates_heap)
            #self.idx_to_be_dismissed.add(idx)
            self.max_loss_candidates = self.evict_candidates_heap[0][0]

        elif -abs(loss) < self.max_loss_candidates:
            return

        else:
            # replace
            popped_loss, popped_idx = heapq.heapreplace(self.evict_candidates_heap, (-abs(loss), idx))
            #self.idx_to_be_dismissed.add(idx)
            self.max_loss_candidates = self.evict_candidates_heap[0][0]

            _ = self.release_from_idx(popped_idx, has_to_delete_idx=False)

            self._cache(idx, sample, target, 0, loss)


    def make_evict_candidates(self, min_reuse_factor, evict_ratio):
        '''
        0. `use_factor update()` for the remaining ones in `self.idx_to_be_dismissed`
        1. Making heap by scanning all the elements in `self.cache_sample`: that `reuse_factor` exceeds min value
            -> heap has (reuse_factor, loss, index) as value
        2. Extract from heap using `heapq.nsmallest(num, q)`
        3. Remove the `reuse_factor` from the heap element tuple
        '''
        #self.update_reuse_factor_for_remain_evicter()
        scan_evict_indices_heap = []    # [ (reuse_factor, abs(loss), index) ]
        for k, v in self.cache_sample.items():
            if (v[2] >= min_reuse_factor):
                #heapq.heappush(scan_evict_indices_heap, (v[2], v[3], k))
                scan_evict_indices_heap.append((v[2], v[3], k))
        evict_candidates_heap = heapq.nlargest(int(self.cache_len * evict_ratio), scan_evict_indices_heap)

        self.evict_candidates_heap = [(-i[1], i[2]) for i in evict_candidates_heap]
        heapq.heapify(self.evict_candidates_heap)
        self.idx_to_be_dismissed = {i[2] for i in evict_candidates_heap}
        self.max_loss_candidates = self.evict_candidates_heap[0][0] if len(self.evict_candidates_heap) > 0 else -(10**9)

        del scan_evict_indices_heap, evict_candidates_heap
        gc.collect()    # invoke garbage collector manually

    def update_reuse_factor_for_remain_evicter(self):
        for idx in self.idx_to_be_dismissed:
            cached_data_element = self.cache_sample[idx]  # (sample, target, reuse_factor, abs(loss))
            cached_data_element[2] += 1
            self.cache_sample[idx] = cached_data_element
        del self.idx_to_be_dismissed

    def release_from_idx(self, idx: int, has_to_delete_idx=False):
        sample, target, reuse_factor, _ = self.cache_sample[idx]

        #sample.release()    # memoryview
        #target.release()    # memoryview
        del sample
        del target

        del self.cache_sample[idx]

        if has_to_delete_idx:
            #self.idx_to_be_dismissed.remove(idx)
            self.idx_to_be_dismissed.discard(idx)   # Some elements that have been replaced in this epoch may not be in the `self.idx_to_be_dismissed`.

        return reuse_factor