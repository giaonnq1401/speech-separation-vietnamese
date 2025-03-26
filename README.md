# Fine-tuning SepFormer-Libri2Mix & Conv-TasNet on ViVOS Dataset with SpeechBrain

## Overview
This repository is a fork of [SpeechBrain](https://github.com/speechbrain/speechbrain) to fine-tune the SepFormer-Libri2Mix and Conv-TasNet models using the [ViVOS dataset](https://github.com/DragonPow/audio_mixing). The goal is to enhance the performance of these models on Vietnamese speech separation tasks.

## Dataset
- **ViVOS**: A Vietnamese speech corpus designed for speech recognition and separation tasks.
  - Dataset repository: [ViVOS GitHub Repository](https://github.com/DragonPow/audio_mixing)
  - Paper: [A non-expert Kaldi recipe for Vietnamese Speech Recognition System](https://aclanthology.org/W16-5207.pdf)
- **LibriMix**: A dataset for training speech separation models, generated from LibriSpeech.
  - Used for training Conv-TasNet.
  - Generated using [JorisCos/LibriMix](https://github.com/JorisCos/LibriMix).
  - Paper: [LibriMix: An Open-Source Dataset for Speech Separation](https://arxiv.org/abs/2005.11262).

## Models Used
- **SepFormer-Libri2Mix**: A state-of-the-art model for speech separation.
  - Paper: [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154).
  - Pre-trained model from SpeechBrain, fine-tuned on ViVOS.
- **Conv-TasNet**: A time-domain speech separation model.
  - Paper: [Conv-TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454).
  - Since SpeechBrain does not provide a pre-trained Conv-TasNet model, it was trained from scratch using the LibriMix dataset before fine-tuning on ViVOS.

## Fine-tuning Process
1. **Data Preparation**
   - **LibriMix Generation**: Used [JorisCos/LibriMix](https://github.com/JorisCos/LibriMix) to create the LibriMix dataset.
   - **ViVOS Processing**: Preprocessed ViVOS to match the required format.

2. **Training**
   - Pre-trained Conv-TasNet on LibriMix.
   - Fine-tuned both SepFormer and Conv-TasNet on ViVOS.

3. **Evaluation**
   - Metrics: SDR, SI-SNR.
   - Comparison with baseline models.

## Modified Files & New Additions
To help reviewers easily find the modifications made to this repository, below is a list of the key files added or modified:

- **Fine-tuning scripts:**
  - `recipes/VIVOS/separation/train.py` → Fine-tuning script for ViVOS dataset.

- **Dataset Processing:**
  - `recipes/VIVOS/separation/prepare_data.py` → Script to convert ViVOS dataset into the required format.
  - `data/LibriMix/create_librimix_from_metadata.py` → Script for generating LibriMix dataset using `JorisCos/LibriMix`.

- **Training Configurations:**
  - `recipes/VIVOS/separation/hparams/sepformer-vivos.yaml` → Configuration for fine-tuning SepFormer.
  - `recipes/LibriMix/separation/hparams/convtasnet.yaml` → Configuration for training Conv-TasNet.
  - `recipes/VIVOS/separation/hparams/convtasnet.yaml` → Configuration for fine-tuning Conv-TasNet.

