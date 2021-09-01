import os


def run_wandb_local():
    os.system("poetry run wandb local --port 8000")


def run_tensorboard_server():
    os.system("poetry run tensorboard --logdir .logs")
