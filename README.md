# Modified in this repo: interfaces for pretrained mmbt, late_fusion, vilbert, visual_bert.

The MMF library provides an initial code base to develop multimodal machine learning methods, but it was difficult to just get those models to work without using CLI as little information was given in their docs. 

Inspired by mmf/models/interfaces/mmbt.py, this repo provides interfaces that supports using pretrained models to do inference on raw images and texts for the hateful memes challenge.

The supported pretrained models are: mmbt, late_fusion, vilbert, visual_bert.

The modified files are:

- mmf/datasets/processors/frcnn_processor.py
- mmf/models/fusions.py
- mmf/models/mmbt.py
- mmf/models/vilbert.py
- mmf/models/visual_bert.py
- mmf/utils/checkpoint.py
- mmf/modules/encoders.py
- tools/scripts/features/frcnn/
- tools/scripts/features/frcnn/
- tools/scripts/features/frcnn/
- tools/scripts/features/frcnn/extract_features_frcnn.py (https://github.com/hivestrung/mmf/commit/6332a9803721a9a230913b6e2589fed172ae4778)
- tools/scripts/features/frcnn/frcnn_utils.py
- tools/scripts/features/frcnn/modeling_frcnn.py
- tools/scripts/features/frcnn/processing_image.py

Replace these files in the mmf code base, and modify tools/scripts/features/frcnn/extract_features_frcnn.py line 28, 29, set the paths to the [feature extractor checkpoints](https://drive.google.com/drive/folders/1Kj6U-YeHrcuuWjbxtYGSttxvoDZVHxyA?usp=sharing). Then you can run inference using pretrained downloadable checkpoints with:

```
from mmf.models.visual_bert import VisualBERT

visual_bert_model = VisualBERT.from_pretrained("visual_bert.finetuned.hateful_memes.direct")
output = visual_bert_model.classify(img, text)  # image should be a PIL Image object, text is a string
```

After downloading the pretrained checkpoints, permission error might occur. To solve this, cd to that checkpoint location, find the config file comes with it, change the cache_dir, data_dir, save_dir etc. to your own paths. 

---

<div align="center">
<img src="https://mmf.sh/img/logo.svg" width="50%"/>
</div>

#

<div align="center">
  <a href="https://mmf.sh/docs">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/mmf">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
  </a>
</div>

---

MMF is a modular framework for vision and language multimodal research from Facebook AI Research. MMF contains reference implementations of state-of-the-art vision and language models and has powered multiple research projects at Facebook AI Research. See full list of project inside or built on MMF [here](https://mmf.sh/docs/notes/projects).

MMF is powered by PyTorch, allows distributed training and is un-opinionated, scalable and fast. Use MMF to **_bootstrap_** for your next vision and language multimodal research project by following the [installation instructions](https://mmf.sh/docs/getting_started/installation). Take a look at list of MMF features [here](https://mmf.sh/docs/getting_started/features).

MMF also acts as **starter codebase** for challenges around vision and
language datasets (The Hateful Memes, TextVQA, TextCaps and VQA challenges). MMF was formerly known as Pythia. The next video shows an overview of how datasets and models work inside MMF. Checkout MMF's [video overview](https://mmf.sh/docs/getting_started/video_overview).


## Installation

Follow installation instructions in the [documentation](https://mmf.sh/docs/getting_started/installation).

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).

## Citation

If you use MMF in your work or use any models published in MMF, please cite:

```bibtex
@misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}
```

## License

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
