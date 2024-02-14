r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.

> `DataLoaderWithCache`:
> Perform the same operation as `DataLoader` of Pytorch using multithread & caching. [Pytorch_version: 2.1]
"""

import itertools

from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union

import torch
import torch.utils.data.graph_settings
from torch.utils.data.datapipes.iter.combinatorics import SamplerIterDataPipe

from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler, BatchSampler
from sampler import BundleRandomSampler, SizedBatchSampler, SequentialSubsetSampler

from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper

from torch.utils.data import _utils

from torch.utils.data.dataloader import _InfiniteConstantSampler, T_co, _collate_fn_t, _worker_init_fn_t, _share_dist_seed, _sharding_worker_init_fn
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter, _DatasetKind

class BundleDataLoader(torch.utils.data.DataLoader):
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = "",
                 bundle_ratio: Optional[float] = None,
                 alternating_order: bool = False):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.')
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError('prefetch_factor option should be non-negative')

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # To manage data in bundles
        if bundle_ratio:
            self.bundle_size = int(round(bundle_ratio * len(self.dataset)))
        else:
            self.bundle_size = None
        self.alternating_order = alternating_order

        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                if shuffle and self.bundle_size:
                    sampler = SamplerIterDataPipe(dataset, sampler=BundleRandomSampler,
                                                  sampler_kwargs={'bundle_size':self.bundle_size, 'generator':generator, 'alternating_order':self.alternating_order})
                else:
                    # See NOTE [ Custom Samplers and IterableDataset ]
                    sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    if not self.bundle_size:
                        sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                    else:
                        sampler = BundleRandomSampler(dataset, bundle_size=self.bundle_size, generator=generator, alternating_order=self.alternating_order)
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

class DataLoaderWithCache(torch.utils.data.DataLoader):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.
    """
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):

        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.')
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError('prefetch_factor option should be non-negative')

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.shuffle = shuffle

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable

            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        self.file_idx = list(self.dataset.imgs_dict.keys())
        self.cache_idx = list(self.dataset.cache_info.keys()) #list(self.cache_dataset.samples.keys())

        self.batch_size = batch_size
        self.batch_num = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        self.drop_last = drop_last
        self.generator = generator
        self.file_sampler = self._file_sampler
        self.cache_sampler = self._cache_sampler
        self.batch_sampler = self._file_batch_sampler
        self.cache_batch_sampler = self._cache_batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

    @property
    def _file_sampler(self):
        # initialize _file_sampler
        if self._dataset_kind == _DatasetKind.Iterable:
            # See NOTE [ Custom Samplers and IterableDataset ]
            return _InfiniteConstantSampler()
        else:  # map-style
            if self.shuffle:
                return SubsetRandomSampler(self.file_idx, generator=self.generator)  # type: ignore[arg-type]
            else:
                return SequentialSubsetSampler(self.file_idx)  # type: ignore[arg-type]

    @property
    def _cache_sampler(self):
        # initialize _cache_sampler
        if self.shuffle and len(self.cache_idx) > 0:
            return SubsetRandomSampler(self.cache_idx, generator=self.generator)
        else:
            return SequentialSubsetSampler(self.cache_idx)

    @property
    def _file_batch_sampler(self):
        return SizedBatchSampler(self._file_sampler, self.batch_num, True)

    @property
    def _cache_batch_sampler(self):
        return SizedBatchSampler(self._cache_sampler, self.batch_num, False)

    def _get_iterator(self) -> '_BaseDataLoaderWithCacheIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderWithCacheIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderWithCacheIter(self)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderWithCacheIter':
        # reset index
        self.dataset.make_evict_candidates()

        self.file_idx = list(self.dataset.imgs_dict.keys())
        self.cache_idx = list(self.dataset.cache_info.keys())

        self.file_sampler = self._file_sampler
        self.cache_sampler = self._cache_sampler

        # When using a single worker the returned iterator should be
        # created everytime to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _file_index_sampler(self):
        if self._auto_collation:
            return self._file_batch_sampler
        else:
            return self._file_sampler

    @property
    def _cache_index_sampler(self):
        if self._auto_collation:
            return self._cache_batch_sampler
        else:
            return self._cache_sampler

class _BaseDataLoaderWithCacheIter(_BaseDataLoaderIter):
    def __init__(self, loader: DataLoaderWithCache) -> None:
        super().__init__(loader)
        self._file_index_sampler = loader._file_index_sampler
        self._cache_index_sampler = loader._cache_index_sampler
        self._file_sampler_iter = iter(self._file_index_sampler)
        self._cache_sampler_iter = iter(self._cache_index_sampler)
        self._sampler_iter = itertools.zip_longest(self._file_sampler_iter, self._cache_sampler_iter)

    def _reset(self, loader, first_iter=False):
        self._file_sampler_iter = iter(self._file_index_sampler)
        self._cache_sampler_iter = iter(self._cache_index_sampler)
        self._sampler_iter = itertools.zip_longest(self._file_sampler_iter, self._cache_sampler_iter)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)

class _SingleProcessDataLoaderWithCacheIter(_BaseDataLoaderWithCacheIter, _SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderWithCacheIter, self).__init__(loader)

class _MultiProcessingDataLoaderWithCacheIter(_MultiProcessingDataLoaderIter, _BaseDataLoaderWithCacheIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderWithCacheIter, self).__init__(loader)