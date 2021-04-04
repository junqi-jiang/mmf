# Copyright (c) Facebook, Inc. and its affiliates.

# Used for MMF internal models for hateful memes task that make predictions
# on raw images

import os
import tempfile
from pathlib import Path
from typing import Type, Union
import numpy as np
import torch
import torchvision.datasets.folder as tv_helpers
from omegaconf import DictConfig
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_processors
from mmf.utils.download import download
from PIL import Image
from torch import nn


ImageType = Union[Type[Image.Image], str]
PathType = Union[Type[Path], str]
BaseModelType = Type[BaseModel]


class FeatureModelInterface(nn.Module):

    def __init__(self, model: BaseModelType, config: DictConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.processor_dict = None
        self.init_processors()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_processors(self):
        config = self.config.dataset_config.hateful_memes
        extra_params = {"data_dir": config.data_dir}
        self.processor_dict = build_processors(config.processors, **extra_params)

    def classify(self, image: ImageType, text: str, zero_image=False, zero_text=False):
        """Classifies a given image and text in it into Hateful/Non-Hateful.
        Image can be a url or a local path or you can directly pass a PIL.Image.Image
        object. Text needs to be a sentence containing all text in the image.

        Args:
            image (ImageType): Image to be classified
            text (str): Text in the image
            zero_image: zero out the image features when classifying
            zero_text: zero out the text features when classifying

        Returns:
            {"label": 0, "confidence": 0.56}
        """
        if isinstance(image, str):
            if image.startswith("http"):
                temp_file = tempfile.NamedTemporaryFile()
                download(image, *os.path.split(temp_file.name), disable_tqdm=True)
                image = tv_helpers.default_loader(temp_file.name)
                temp_file.close()
            else:
                image = tv_helpers.default_loader(image)

        text = self.processor_dict["text_processor"]({"text": text})
        # image = self.processor_dict["image_processor"](image)
        im_feature_0 = np.load(
            "/Users/JQJiang/Desktop/explainable-multimodal-classification/multimodal/hateful-memes/LIME/extract-feature/feat/gun.npy",
            allow_pickle=True)
        im_feature_0 = torch.from_numpy(im_feature_0)

        sample = Sample()
        sample.text = text["text"]
        if "input_ids" in text:
            sample.update(text)

        # load image_info:
        im_info_0 = np.load(
            "/Users/JQJiang/Desktop/explainable-multimodal-classification/multimodal/hateful-memes/LIME/extract-feature/feat/gun_info.npy",
            allow_pickle=True)
        sample_im_info = Sample()
        sample_im_info.bbox = im_info_0[()]['bbox']
        sample_im_info.num_boxes = im_info_0[()]['num_boxes']
        sample_im_info.image_width = im_info_0[()]['image_width']
        sample_im_info.image_height = im_info_0[()]['image_height']
        sample_list_info = SampleList([sample_im_info])

        sample.image_feature_0 = im_feature_0
        sample.image_info_0 = sample_list_info
        sample_list = SampleList([sample])
        device = next(self.model.parameters()).device
        sample_list = sample_list.to(device)

        output = self.model(sample_list)
        scores = nn.functional.softmax(output["scores"], dim=1)
        confidence, label = torch.max(scores, dim=1)

        return {"label": label.item(), "confidence": confidence.item()}

