import gc
import os

import numpy as np
import pandas as pd
import scipy
import torch
import wandb

from src.training.data_preparation import construct_data_loaders, load_data
from src.training.models import Seq2one_model
from src.training.training_script import evaluate_model
from visualization.ConfusionMatrixVisualization import plot_and_save_confusion_matrix


def get_info_and_download_models_weights_from_project(entity: str, project_name: str, output_path: str) -> pd.DataFrame:
    """ Extracts info about run models from the project and downloads the models weights to the output_path.
        The extracted information will be stored as pd.DataFrame with the columns:
        ['ID', 'model_type', 'window_size', 'stride' 'loss_multiplication_factor', 'best_val_recall']

    :param entity: str
            The name of the WandB entity. (usually account name)
    :param project_name: str
            The name of the WandB project.
    :param output_path: str
            The path to the folder where the models weights will be downloaded.
    :return: pd.DataFrame
            The extracted information about the models.
    """
    # get api
    api = wandb.Api()
    # establish the entity and project name
    entity, project = entity, project_name
    # get runs from the project
    runs = api.runs(f"{entity}/{project}")
    # extract info about the runs
    info = pd.DataFrame(columns=['ID', 'model_type', 'window_size', 'stride',
                                 'loss_multiplication_factor', 'best_val_recall'])
    for run in runs:
        # check if the model was saved during training. If not, skip this run. It can be either because of the error
        # or because it is in the middle of training
        ID = run.name
        model_type = run.config['MODEL_TYPE']
        window_size = run.config['window_size']
        stride = run.config['stride']
        loss_multiplication_factor = run.config['loss_multiplication_factor']
        best_val_recall = run.config['best_val_recall']
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'model_type': [model_type],
                               'window_size': [window_size],
                               'stride': [stride],
                               'loss_multiplication_factor': [loss_multiplication_factor],
                               'best_val_recall': [best_val_recall]}
                          )
                          ]
                         )
        # download the model weights
        final_output_path = os.path.join(output_path, ID)
        run.file('best_model_general_recall.pth').download(final_output_path, replace=True)
        # move the file out of dir and rename file for convenience
        os.rename(os.path.join(final_output_path, 'best_model_general_recall.pth'),
                  final_output_path + '.pth')
        # delete the dir
        os.rmdir(final_output_path)

    return info

def draw_confusion_matrix(model, generator, device, output_path, filename):
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
            outputs = model(inputs)  # list of outputs for each classification task

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

    plot_and_save_confusion_matrix(y_true=ground_truth[:, 0], y_pred=predictions[:, 0], name_labels=['0', '1', '2'],
    path_to_save = output_path, name_filename = '%s_q1.png' % filename)

    plot_and_save_confusion_matrix(y_true=ground_truth[:, 1], y_pred=predictions[:, 1], name_labels=['0', '1', '2'],
                                   path_to_save=output_path, name_filename='%s_q2.png' % filename)

    plot_and_save_confusion_matrix(y_true=ground_truth[:, 2], y_pred=predictions[:, 2], name_labels=['0', '1', '2'],
                                   path_to_save=output_path, name_filename='%s_q3.png' % filename)









def main():
    # params
    path_to_train_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/train_embeddings.csv"
    path_to_dev_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/dev_embeddings.csv"
    path_to_test_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/test_embeddings.csv"
    output_path = "/nfs/home/ddresvya/scripts/BEA/models"
    print("Start of the script....")
    # load data
    print("Loading data....")
    data = load_data(paths_to_dfs=[path_to_train_df, path_to_dev_df, path_to_test_df])
    # load all model weights
    print("Downloading models....")
    info = get_info_and_download_models_weights_from_project(entity='denisdresvyanskiy', project_name='BEA_project',
                                                                output_path='/nfs/home/ddresvya/scripts/BEA/models')
    info = info.reset_index()
    # add columns to the info
    info['0_test_recall'] = None
    info['1_test_recall'] = None
    info['2_test_recall'] = None
    info['0_dev_recall'] = None
    info['1_dev_recall'] = None
    info['2_dev_recall'] = None
    # evaluate every model in info
    print("Evaluating models....")
    for i in range(info.shape[0]):
        # get model type
        model_type = info['model_type'].iloc[i]
        window_size = float(info['window_size'].iloc[i])
        stride = float(info['stride'].iloc[i])
        # construct data loaders
        train_generator, dev_generator, test_generator = construct_data_loaders(video_dicts=data,
                                                                                label_columns=['q1', 'q2', 'q3'],
                                                                                feature_columns=["emb_%i" % i for i in
                                                                                                 range(256)],
                                                                                window_size=window_size,
                                                                                stride=stride, batch_size=64)
        # construct model
        if model_type == 'Transformer-Based-1-block':
            model = Seq2one_model(input_size=256, num_classes=[3,3,3], transformer_num_heads=4,
                          num_timesteps=int(window_size*5), num_transformer_layers=1)
        elif model_type == 'Transformer-Based-2-block':
            model = Seq2one_model(input_size=256, num_classes=[3,3,3], transformer_num_heads=16,
                          num_timesteps=int(window_size*5), num_transformer_layers=2)
        else:
            raise ValueError("Wrong model type.")
        # load model weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # test model
        test_metrics = evaluate_model(model, test_generator, device)
        draw_confusion_matrix(model, test_generator, device, output_path=os.path.join(output_path, 'cm'), filename=info['ID'].iloc[i]+'_test')
        dev_metrics = evaluate_model(model, dev_generator, device)
        draw_confusion_matrix(model, dev_generator, device, output_path=os.path.join(output_path, 'cm'), filename=info['ID'].iloc[i]+'_dev')
        # change the prefix for test metrics
        test_metrics = [{metric_name.replace('val','test'):metric_value for metric_name, metric_value in metrics_task.items()} for metrics_task in test_metrics]
        # add metrics to the info
        info.loc[i, '0_test_recall'] = test_metrics[0]['0_test_recall']
        info.loc[i, '1_test_recall'] = test_metrics[1]['1_test_recall']
        info.loc[i, '2_test_recall'] = test_metrics[2]['2_test_recall']
        info.loc[i, '0_dev_recall'] = dev_metrics[0]['0_val_recall']
        info.loc[i, '1_dev_recall'] = dev_metrics[1]['1_val_recall']
        info.loc[i, '2_dev_recall'] = dev_metrics[2]['2_val_recall']

        # save info
        info.to_csv('/nfs/home/ddresvya/scripts/BEA/models/info.csv', index=False)

        # clear memory
        del model, train_generator, dev_generator, test_generator
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

