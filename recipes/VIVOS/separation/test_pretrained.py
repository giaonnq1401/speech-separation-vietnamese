#!/usr/bin/env/python3
"""
Script for testing a pretrained speech separation model on VIVOS dataset
without fine-tuning.
"""

import os
import sys
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

# Import the separation class from train.py
# Make sure train.py is in the same directory
from train import Separation, dataio_prep

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP testing)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # Create dataset objects (only need test data)
    _, test_data = dataio_prep(hparams)

    # Make sure pretrained_separator is specified in the yaml file
    if "pretrained_separator" not in hparams:
        raise ValueError("pretrained_separator must be specified in the hparams file")

    # Load pretrained model
    run_on_main(hparams["pretrained_separator"].collect_files)
    hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Only evaluation
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
    
    print("Testing completed. Results saved in:", hparams["output_folder"])