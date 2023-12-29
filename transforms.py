import torch
from typing import Dict, Type, Union, Any
import copy

from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import AutoAugmentPolicy, InterpolationMode
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType

class AutoAugment_with_p(transforms.AutoAugment):
    r"""
    Args:
        policy (AutoAugmentPolicy, optional): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        p (float): Probability of applying transform among all data samples.
    """
    def __init__(self,
                 policy = AutoAugmentPolicy.IMAGENET,
                 interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
                 fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
                 p: float = 1.0):
        super().__init__(policy=policy, interpolation=interpolation, fill=fill)
        self.p = p

    def forward(self, *inputs):
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        if torch.rand(()) > self.p:
            return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue

            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]

            magnitudes = magnitudes_fn(10, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            image_or_video = self._apply_image_or_video_transform(
                image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill
            )

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


class RandAugment_with_p(transforms.RandAugment):
    r"""
    Args:
        num_ops (int, optional): Number of augmentation transformations to apply sequentially.
        magnitude (int, optional): Magnitude for all the transformations.
        num_magnitude_bins (int, optional): The number of different magnitude values.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        p (float): Probability of applying transform among all data samples.
    """
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
        p: float = 1.0
    ) -> None:
        super().__init__(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins,
                         interpolation=interpolation, fill=fill)
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        if torch.rand(()) > self.p:
            return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
            magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[self.magnitude])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image_or_video = self._apply_image_or_video_transform(
                image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill
            )

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)

# ------
basic_transform = transforms.Compose([
  transforms.Resize((224,224)),   #transforms.RandomCrop(224, padding=4),
  #transforms.RandomHorizontalFlip(),
  transforms.PILToTensor(), #transforms.ToTensor(),
  transforms.ToDtype(torch.float32, scale=True),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

valset_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.PILToTensor(),
  transforms.ToDtype(torch.float32, scale=True),
])

# ------
def autoaugment_transform(basic_transform, p=1):
    autoaugment_transform = copy.deepcopy(basic_transform)
    transform_block = AutoAugment_with_p(policy=transforms.AutoAugmentPolicy.CIFAR10, p=p)

    # Add AutoAugment at first
    autoaugment_transform.transforms.insert(0, transform_block)
    
    return autoaugment_transform, transform_block

# ------
def randaugment_transform(basic_transform, num_ops, p=1):
    randaugment_transform = copy.deepcopy(basic_transform)
    transform_block = RandAugment_with_p(num_ops=num_ops, p=p)

    # Add RandAugment at first
    randaugment_transform.transforms.insert(0, transform_block)
    
    return randaugment_transform, transform_block