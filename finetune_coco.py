""" Example on how to finetune on COCO dataset
"""
import tensorflow as tf

from detr_tf.data.coco import load_coco_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training

try:
    # Should be optional if --log is not set
    import wandb
except:
    wandb = None


import time


def build_model(config):
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load the pretrained model
    detr = get_detr_model(config, include_top=True, weights="detr")
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset
    train_dt, coco_class_names = load_coco_dataset("train", config.batch_size, config, augmentation=True)
    valid_dt, _ = load_coco_dataset("val", 1, config, augmentation=False)

    # Train/finetune the transformers only
    config.train_backbone = False
    config.train_transformers = True

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    # Run the training for 5 epochs
    for epoch_nb in range(100):
        training.eval(detr, valid_dt, config, coco_class_names, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, coco_class_names)


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    if config.log:
        wandb.init(project="detr-tensorflow", reinit=True)

    # Run training
    run_finetuning(config)
