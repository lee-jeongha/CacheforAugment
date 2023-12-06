import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler

class BatchWithCacheSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        cached_index (List[int] or Iterable[int]): Cached index.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], cached_index: Union[List[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.is_first_iter = True
        if cached_index is not None:
            self.len_cached_index = len(cached_index)
            self.cached_index = cached_index[torch.randperm(self.len_cached_index, generator=torch.Generator())]
        else:
            self.len_cached_index = 0
            self.cached_index = None
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        cached_num_per_batch = self.len_cached_index // self.__len__()
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            mv_iter = iter(self.cached_index)
            if self.cached_index is not None:
                while True:
                    try:
                        while len(batch) == (self.batch_size - cached_num_per_batch):
                            len_curr_batch = len(curr_batch) if curr_batch is not None else 0
                            curr_batch = [next(sampler_iter) for _ in range(self.batch_size - len_curr_batch - cached_num_per_batch)]
                            curr_batch = [item for item in batch if item not in self.cached_index]
                            batch += curr_batch
                        batch += [next(mv_iter) for _ in range(cached_num_per_batch)]
                        yield batch
                    except StopIteration:
                        break
            else:
                while True:
                    try:
                        batch = [next(sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                    except StopIteration:
                        break
        
        else:
            if self.cached_index is not None:
                batch = [self.cached_index[:cached_num_per_batch]] + [0] * (self.batch_size - cached_num_per_batch)
                base_in_mv = cached_num_per_batch
                idx_in_batch = cached_num_per_batch

                for idx in self.sampler:
                    if idx in self.cached_index:
                        continue                
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        base_in_mv += cached_num_per_batch
                        batch = [self.cached_index[base_in_mv:base_in_mv+cached_num_per_batch]] + [0] * (self.batch_size - cached_num_per_batch)
                        idx_in_batch = cached_num_per_batch

                if idx_in_batch > cached_num_per_batch:    # end of sampler
                    if base_in_mv < self.len_cached_index:   # but still has memoryview_index left
                        batch = batch[:idx_in_batch] + self.cached_index[base_in_mv:]
                        idx_in_batch += (self.len_cached_index - base_in_mv)
                        if idx_in_batch > self.batch_size:
                            for _ in range(idx_in_batch // self.batch_size):
                                yield batch[:self.batch_size]
                                batch = batch[self.batch_size:]
                            yield batch
                    else:
                        yield batch[:idx_in_batch]
            
            else:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for idx in self.sampler:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                if idx_in_batch > 0:
                    yield batch[:idx_in_batch]
        
    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class BundleRandomSampler(Sampler[int]):
    r"""Samples elements randomly in a bundle size.

    Args:
        data_source (Dataset): dataset to sample from
        bundle_size (int): size of bundle
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, data_source: Sized, bundle_size: int,
                 num_samples: Optional[int] = None, generator=None, alternating_order=True) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.bundle_size = bundle_size
        self.offsets = torch.arange(0, len(self.data_source), self.bundle_size)
        self.shuffled_list = torch.randperm(self.num_samples, generator=self.generator)
        self.alternating_order = alternating_order
        self.in_order = True

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.in_order:
            self.offsets = torch.arange(0, len(self.data_source), self.bundle_size)
        else:
            #self.offsets = torch.flip(torch.arange(0, len(self.data_source), self.bundle_size), dims=[0])
            self.offsets = torch.cat((torch.arange(len(self.data_source)-self.bundle_size, 0, -self.bundle_size), torch.zeros([1, ], dtype=torch.int32)), dim=0)

        for i in range(self.num_samples // self.bundle_size):
            yield from self.shuffled_list[torch.add(torch.randperm(self.bundle_size, generator=generator), self.offsets[i])]#.tolist()
            #yield from torch.add(torch.randperm(self.bundle_size, generator=generator), self.offsets[i]).tolist()
            #yield from torch.randperm(self.bundle_size, generator=generator).tolist()
        if (self.num_samples % self.bundle_size):
            yield from self.shuffled_list[torch.add(torch.randperm(self.num_samples % self.bundle_size, generator=generator), self.offsets[-1])]#.tolist()
            #yield from torch.add(torch.randperm(self.num_samples % self.bundle_size, generator=generator), self.offsets[-1]).tolist()
            #yield from torch.randperm(self.num_samples % self.bundle_size, generator=generator).tolist()[:self.num_samples % self.bundle_size]

        if self.alternating_order:
            self.in_order = not(self.in_order)


    def __len__(self) -> int:
        return self.num_samples