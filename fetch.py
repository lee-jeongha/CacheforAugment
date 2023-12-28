r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.

> `_MultithreadBaseDatasetFetcher`, `_MultithreadIterableDatasetFetcher`, `_MultithreadMapDatasetFetcher`:
> Perform the same operation as `_BaseDatasetFetcher`, `_IterableDatasetFetcher`, `_MapDatasetFetcher` of Pytorch using multithread. [Pytorch_version: 2.1]
"""

from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data._utils.fetch import _BaseDatasetFetcher

class _MultithreadBaseDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, num_threads : Optional[int] = None):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_threads = num_threads

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _MultithreadIterableDatasetFetcher(_MultithreadBaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, num_threads):
        super().__init__(dataset, auto_collation, collate_fn, drop_last, num_threads)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def single_fetch(self):
        return next(self.dataset_iter)

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                exes = [executor.submit(self.single_fetch)]
                data += [exe.result() for exe in exes]
            self.ended = True

            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MultithreadMapDatasetFetcher(_MultithreadBaseDatasetFetcher):
    def single_fetch(self, idx):
        return self.dataset[idx]

    def fetch(self, possibly_batched_index):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            exes = [executor.submit(self.single_fetch, idx) for idx in possibly_batched_index]
            data = [exe.result() for exe in exes]

        return self.collate_fn(data)