import torch
from typing import Dict, Type, Union

from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import AutoAugmentPolicy, InterpolationMode
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType

class MyAutoAugment(transforms.AutoAugment):
    def __init__(self,
                 policy = AutoAugmentPolicy.IMAGENET,
                 interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
                 fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
                 num_ops = 0):
        super().__init__(policy=policy, interpolation=interpolation, fill=fill)
        self.num_ops = num_ops

    def forward(self, *inputs):
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)
        op_count = 0

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            #if not torch.rand(()) <= probability:
            #    continue

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

            # count the number of augmentation layer
            op_count += 1
            if op_count == self.num_ops:
                break

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)

# ------
basic_transform = transforms.Compose([
  transforms.Resize((224,224)),   #transforms.RandomCrop(224, padding=4),
  #transforms.RandomHorizontalFlip(),
  transforms.PILToTensor(), #transforms.ToTensor(),
  transforms.ToDtype(torch.float32),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ------
def autoaugment_transform(basic_transform, num_ops):
    autoaugment_transform = basic_transform
    transform_block = MyAutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10, num_ops=num_ops)

    # Add AutoAugment at first
    autoaugment_transform.transforms.insert(0, transform_block)
    
    return autoaugment_transform, transform_block

# ------
def randaugment_transform(basic_transform, num_ops):
    randaugment_transform = basic_transform
    transform_block = transforms.RandAugment(num_ops=num_ops)

    # Add RandAugment at first
    randaugment_transform.transforms.insert(0, transform_block)
    
    return randaugment_transform, transform_block