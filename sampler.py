import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler

class SizedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_num (int): Number of mini-batch.

    Example:
        >>> list(SizedBatchSampler(SequentialSampler(range(10)), batch_num=3, upper_first=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        >>> list(SizedBatchSampler(SequentialSampler(range(10)), batch_num=3, upper_first=True))
        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_num: int, upper_first: bool = True) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if batch_num is None:
            raise ValueError(f"batch_num should be a positive integer value, but got batch_num={batch_num}")

        self.sampler = sampler
        self.lower_batch_size = len(self.sampler) // batch_num
        self.upper_batch_size = self.lower_batch_size + 1
        self.batch_num = batch_num
        self.upper_batch_num = len(self.sampler) % batch_num
        self.lower_batch_num = batch_num - self.upper_batch_num
        self.upper_first = upper_first

        if (not self.upper_first) or (self.upper_batch_num == 0):
            self.batch_nums = [self.lower_batch_num, self.upper_batch_num]
            self.batch_sizes = [self.lower_batch_size, self.upper_batch_size]
        else:
            self.batch_nums = [self.upper_batch_num, self.lower_batch_num]
            self.batch_sizes = [self.upper_batch_size, self.lower_batch_size]

        self.batch_counter = 0

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self._batch_size <= 0:
            return

        batch = [0] * self._batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            if self.batch_counter > self.batch_num:
                raise StopIteration
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self._batch_size:
                yield batch
                self.batch_counter += 1
                idx_in_batch = 0
                batch = [0] * self._batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]
            self.batch_counter += 1

    @property
    def _batch_size(self):
        if self.batch_counter >= self.batch_nums[0]:
            return self.batch_sizes[1]
        else:
            return self.batch_sizes[0]

    def __len__(self) -> int:
        return self.batch_num


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