import subprocess
import os
import yaml
import click


def load_config():
    with open('config.yml') as f:
        return yaml.load(f)


def safe_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path