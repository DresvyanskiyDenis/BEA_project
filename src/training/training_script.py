import sys

import scipy

sys.path.append('/work/home/dsu/BEA_project/')
sys.path.append('/work/home/dsu/datatools/')

import argparse
from torchinfo import summary
import gc
import os
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import wandb
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping
from pytorch_utils.training_utils.losses import SoftFocalLoss
from src.training.data_preparation import load_data, construct_data_loaders, compute_class_weights
from src.training.models import Seq2one_model


def construct_model(num_classes: List[int], num_timesteps: int) -> torch.nn.Module:
    model = Seq2one_model(input_size=256, num_classes=num_classes, transformer_num_heads=4,
                          num_timesteps=num_timesteps)
    return model

def transform_labels_to_one_hot(labels: torch.Tensor, num_classes:int) -> torch.Tensor:
    # input shape (batch_size, sequence_length, number_of_class)
    # Warning: in this case, the number_of_class can be 3-dimensional, for example.
    # This means that the task is multi-task or multi-label classification
    # output shape (batch_size, task, one_hot_encoded)
    labels = labels.cpu().numpy()
    # calculate mode for each task
    labels = scipy.stats.mode(labels, axis=1, keepdims=False)[0]
    # one-hot encode
    labels = [np.eye(num_classes)[labels[:,i].astype(int)] for i in range(labels.shape[1])]
    labels = [torch.from_numpy(label).float() for label in labels]
    # concatenate them back so that they have shape (batch_size, task, one_hot_encoded)
    labels = torch.stack(labels, dim=1)
    return labels




def evaluate_model(model: torch.nn.Module, generator: torch.utils.data.DataLoader, device: torch.device) -> List[Dict[
    object, float]]:
    evaluation_metrics_classification = {'val_accuracy': accuracy_score,
                                         'val_precision': partial(precision_score, average='macro'),
                                         'val_recall': partial(recall_score, average='macro'),
                                         'val_f1': partial(f1_score, average='macro')
                                         }

    # create arrays for predictions and ground truth labels
    predictions = []
    ground_truth = []

    # start evaluation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(generator):
            # get the inputs; data is a list of [inputs, labels]
            # labels are the list of figures, for example, [1, 2, 0]. Those are for three different classification tasks
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            # forward pass
            outputs = model(inputs) # list of outputs for each classification task

            # labels to numpy
            labels = labels.cpu().numpy().squeeze()
            # take mode to get rid of sequence dimension
            labels = scipy.stats.mode(labels, axis=1, keepdims=False)[0]

            # softmax, transformation to numpy, argmax for each classification task
            outputs = [torch.softmax(output, dim=-1) for output in outputs]
            outputs = [output.cpu().numpy().squeeze() for output in outputs]
            outputs = [np.argmax(output, axis=-1) for output in outputs]
            # stack it so that it will have shape (batch_size, task)
            outputs = np.stack(outputs, axis=1)

            # save ground_truth labels and predictions in arrays to calculate metrics afterwards by one time
            predictions.append(outputs)
            ground_truth.append(labels)

        # concatenate all predictions and ground truth labels. Remember that every element os the predictions list
        # is also a list with predictions for each classification task
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)


        # calculate evaluation metrics for each task
        metric_tasks = []
        for task in range(predictions.shape[1]):
            results = {}
            for name, metric in evaluation_metrics_classification.items():
                results[str(task)+"_"+name] = metric(ground_truth[:,task], predictions[:,task])
            metric_tasks.append(results)
        # print evaluation metrics
        print('Evaluation metrics')
        for task_idx in range(len(metric_tasks)):
            print('Task %d' % task_idx)
            for metric_name, metric_value in metric_tasks[task_idx].items():
                print("%s: %.4f" % (metric_name, metric_value))
    # clear RAM from unused variables
    del inputs, labels, outputs, predictions, ground_truth
    torch.cuda.empty_cache()
    return metric_tasks


def train_step(model: torch.nn.Module, criterion: List[torch.nn.Module],
               inputs: Tuple[torch.Tensor, ...], ground_truth: List[torch.Tensor],
               device: torch.device) -> List:
    """ Performs one training step for a model.

    :param model: torch.nn.Module
            Model to train.
    :param criterion: List[torch.nn.Module]
            Loss functions for each output of the model.
    :param inputs: Tuple[torch.Tensor,...]
            Inputs for the model.
    :param ground_truth: torch.Tensor
            Ground truths for the model. SHould be passed as one-hot encoded tensors
    :param device: torch.device
            Device to use for training.
    :return:
    """
    # forward pass
    output = model(inputs) # list of outputs for each classification task
    # calculate loss for every output of the model
    losses = []
    for i in range(len(output)):
        losses.append(criterion[i](output[i], ground_truth[:, i]))
    # sum all losses and average them
    loss = torch.stack(losses).sum() / len(losses)

    # clear RAM from unused variables
    del output, ground_truth

    return [loss]


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: List[torch.nn.Module],
                device: torch.device, print_step: int = 100,
                accumulate_gradients: Optional[int] = 1,
                warmup_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor: Optional[float] = None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterions: List[torch.nn.Module]
            Loss functions for each output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param warmup_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have warmup lr scheduller. In that case, the learning rate is being changed
            after every mini-batch, therefore should be passed to this function.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.to(device)

        # tranfrorm labels to one-hot encoded tensors. Remember that labels is a list of torch.Tensor
        # where each list element is a tensor with labels for each classification task
        labels = transform_labels_to_one_hot(labels, num_classes=3)
        labels = labels.to(device)

        # do train step
        with torch.set_grad_enabled(True):
            # form indices of labels which should be one-hot encoded
            step_losses = train_step(model, criterions, inputs, labels, device)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_lr_scheduller is not None:
                    warmup_lr_scheduller.step()

        # print statistics
        running_loss += sum_losses.item()
        total_loss += sum_losses.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
        # clear RAM from all the intermediate variables
        del inputs, labels, step_losses, sum_losses
    # clear RAM at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / counter


def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                window_size: float, stride: float, consider_timestamps: bool,
                class_weights: List[torch.Tensor],BATCH_SIZE: int, ACCUMULATE_GRADIENTS: int,
                loss_multiplication_factor: Optional[float] = None) -> None:
    print("Start of the model training.")
    # metaparams
    metaparams = {
        # general params
        "architecture": "Transformer-Based",
        "MODEL_TYPE": "Transformer-Based",
        "dataset": "BEA",
        "BEST_MODEL_SAVE_PATH": "best_models/",
        "NUM_WORKERS": 8,
        # temporal params
        "window_size": window_size,
        "stride": stride,
        "consider_timestamps": consider_timestamps,
        # model architecture
        "NUM_CLASSES": [3, 3, 3],
        # training metaparams
        "NUM_EPOCHS": 100,
        "BATCH_SIZE": BATCH_SIZE,
        "OPTIMIZER": "AdamW",
        "EARLY_STOPPING_PATIENCE": 100,
        "WEIGHT_DECAY": 0.0001,
        # LR scheduller params
        "LR_SCHEDULLER": "Warmup_cyclic",
        "ANNEALING_PERIOD": 5,
        "LR_MAX_CYCLIC": 0.005,
        "LR_MIN_CYCLIC": 0.0001,
        "LR_MIN_WARMUP": 0.00001,
        "WARMUP_STEPS": 100,
        "WARMUP_MODE": "linear",
        # loss params
        "loss_multiplication_factor": loss_multiplication_factor,
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="BEA_project", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH': wandb.run.dir}, allow_val_change=True)
    # get one iteration of train generator to get sequence length
    inputs, labels = next(iter(train_generator))
    sequence_length = inputs.shape[1]

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct sequence-to-one model out of base model
    model = construct_model(num_classes=config.NUM_CLASSES, num_timesteps=sequence_length)
    model = model.to(device)
    # print model architecture
    summary(model, (2, sequence_length, 256))

    # select optimizer
    model_parameters = model.parameters()
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # Loss functions
    class_weights = [class_weight.to(device) for class_weight in class_weights]
    criterion = [SoftFocalLoss(softmax=True, alpha=class_weights[0], gamma=2),
                 SoftFocalLoss(softmax=True, alpha=class_weights[1], gamma=2),
                 SoftFocalLoss(softmax=True, alpha=class_weights[2], gamma=2)]
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ANNEALING_PERIOD,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.ANNEALING_PERIOD,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator) // ACCUMULATE_GRADIENTS,
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
    # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
    if config.LR_SCHEDULLER == 'Warmup_cyclic':
        optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # early stopping
    best_val_recall = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterion, device, print_step=100,
                                 accumulate_gradients=ACCUMULATE_GRADIENTS,
                                 warmup_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics = evaluate_model(model, dev_generator, device)
        general_val_metric = (val_metrics[0]["0_val_recall"] + val_metrics[1]["1_val_recall"] + val_metrics[2]["2_val_recall"])/3.

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better recall for all three classification tasks
        if general_val_metric > best_val_recall:
            best_val_recall = general_val_metric
            wandb.config.update({'best_val_recall': best_val_recall}, allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_general_recall.pth'))

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics[0], commit=False)
        wandb.log(val_metrics[1], commit=False)
        wandb.log(val_metrics[2], commit=False)
        wandb.log({'general_val_recall': general_val_metric}, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(general_val_metric)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()

        # check early stopping
        early_stopping_result = early_stopping_callback(general_val_metric, model)
        if early_stopping_result:
            print("Early stopping")
            break
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(window_size, stride, batch_size, accumulate_gradients,
         loss_multiplication_factor):
    # params
    path_to_train_df = "/Data/extracted_embeddings/train_embeddings.csv"
    path_to_dev_df = "/Data/extracted_embeddings/dev_embeddings.csv"
    path_to_test_df = "/Data/extracted_embeddings/test_embeddings.csv"
    print("Start of the script....")
    # load data
    print("Loading data....")
    data = load_data(paths_to_dfs=[path_to_train_df, path_to_dev_df, path_to_test_df])
    # create data loaders
    print("Creating data loaders....")
    train_generator, dev_generator, test_generator = construct_data_loaders(video_dicts= data,
                           label_columns=['q1', 'q2', 'q3'],
                           feature_columns=["emb_%i" % i for i in range(256)],
                           window_size=window_size,
                           stride=stride, batch_size=batch_size)

    # compute class weights
    class_weights = compute_class_weights(data[0])
    # transform class_weights to torch.tensor
    class_weights = [torch.from_numpy(class_w) for class_w in class_weights]

    # train the model
    train_model(window_size=window_size, stride=stride, consider_timestamps=True,
                train_generator=train_generator, dev_generator=dev_generator, class_weights=class_weights,
                BATCH_SIZE=batch_size, ACCUMULATE_GRADIENTS=accumulate_gradients,
                loss_multiplication_factor=loss_multiplication_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Emotion Recognition model training')
    parser.add_argument('--window_size', type=float, required=True)
    parser.add_argument('--stride', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--accumulate_gradients', type=int, required=True)
    parser.add_argument('--loss_multiplication_factor', type=float, required=False, default=1.0)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ", args)
    # check arguments
    if args.batch_size < 1:
        raise ValueError("batch_size should be greater than 0")
    if args.accumulate_gradients < 1:
        raise ValueError("accumulate_gradients should be greater than 0")
    # convert args to bool
    batch_size = args.batch_size
    accumulate_gradients = args.accumulate_gradients
    loss_multiplication_factor = args.loss_multiplication_factor
    window_size = args.window_size
    stride = args.stride
    # run main script with passed args
    main(window_size=window_size, stride=stride,
         batch_size=batch_size, accumulate_gradients=accumulate_gradients,
         loss_multiplication_factor=loss_multiplication_factor)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()
