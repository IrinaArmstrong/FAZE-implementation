# Basic
import os
import sys
import random
from pathlib import Path
from typing import (List, NoReturn, Tuple, Union, Dict, Any)
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             classification_report, f1_score,
                             recall_score, precision_score)
import torch
import torch.nn as nn

import logging_handler
logger = logging_handler.get_logger(__name__)

def check_cuda_available():
    if torch.cuda.is_available():
        logger.info(f"CUDA device is available.")
        return True
    logger.info(f"CUDA device is not available. Change to CPU.")
    return False


def acquire_device(device_type: str = 'cpu'):
    """
    Check available devices and select one.
    """
    device = torch.device(device_type)
    logger.info(f"Preferred device: {device.type}")

    if device.type == 'gpu':
        if not check_cuda_available():
            logger.error(f"Device provided is CUDA, but it is available. Change to CPU.")
            device = torch.device('cpu')
        else:
            logger.info(torch.cuda.get_device_name(0))
            logger.info('Memory Usage, Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            logger.info('Memory Usage, Cached:', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')



def seed_everything(seed_value: int):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # for using CUDA backend
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # get rid of nondeterminism
        torch.backends.cudnn.benchmark = True


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Invalid data type {}'.format(type(data)))


def init_model(model: nn.Module, parameters: Dict[str, Any],
               dir: str='models_checkpoints',
               filename: str='model.pt') -> nn.Module:
    """
    Initialize model and load state dict.
    """
    model = model(**parameters.get("model_params"))
    _ = load_model(model, dir=dir,  filename=filename)
    logger.info(model)
    return model


def save_model(model: nn.Module, save_dir: str, filename: str):
    """
    Trained model, configuration and tokenizer,
    they can then be reloaded using `from_pretrained()` if using default names.
    - save_dir - full path for directory where it is chosen to save model
    - filename - unique file name for model weights
    """
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model.state_dict()
    torch.save(model_to_save, os.path.join(save_dir, filename))
    logger.info("Model successfully saved.")


def load_model(model, dir: Union[str, Path],
               filename: str):
    """
    Loads a model’s parameter dictionary using a deserialized state_dict.
    :param model: model instance (uninitialized)
    :param dir: folder/_path - absolute!
    :param filename: state_dict filename
    :return: initialized model
    """
    if type(dir) == str:
        dir = Path(dir)
    if not dir.resolve().exists():
        logger.error(f"Provided dir do not exists: {dir}!")
        return
    fn = dir / filename
    if not fn.exists():
        logger.error(f"Provided model filename do not exists: {fn}!")
        return
    model.load_state_dict(torch.load(str(fn)))


def compute_metrics(true_labels: List[int],
                    pred_labels: List[int]):

    logger.info("***** Eval results *****")
    ac = accuracy_score(true_labels, pred_labels)
    bac = balanced_accuracy_score(true_labels, pred_labels)

    logger.info('Accuracy score:', ac)
    logger.info('Balanced_accuracy_score:', bac)
    logger.info(classification_report(true_labels, pred_labels))


def clear_logs_dir(dir_path: str, ignore_errors: bool = True):
    """
    Reset logging directory with deleting all files inside.
    """
    dir_path = Path(dir_path).resolve()
    if dir_path.exists():
        files = len(list(dir_path.glob("*")))
        shutil.rmtree(str(dir_path), ignore_errors=ignore_errors)
        logger.info(f"Folder {str(dir_path)} cleared, deleted {files} files.")

    dir_path.mkdir(exist_ok=True)
    logger.info(f"Logging directory created at: {str(dir_path)}")