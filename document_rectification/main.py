import os


def run_wandb_local():
    os.system("poetry run wandb local")


def run_tensorboard_server():
    os.system("poetry run tensorboard --logdir .logs")
