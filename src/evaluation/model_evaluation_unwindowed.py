import gc
import os

import numpy as np
import pandas as pd
import scipy
import torch
import wandb

from src.training.unwindowed.data_preparation import load_data, create_data_generators
from src.training.unwindowed.models import Seq2one_model_unwindowed
from src.training.unwindowed.training_script import evaluate_model
from visualization.ConfusionMatrixVisualization import plot_and_save_confusion_matrix

average_padding= {'rose-microwave-146', 'still-resonance-147', 'dark-shadow-148', 'sweet-river-149', 'stellar-lion-150', 'jolly-thunder-151', 'olive-salad-152', 'rural-bird-153'}
max_padding={'fresh-glitter-154', 'firm-glade-155', 'fearless-star-156', 'wobbly-valley-157', 'good-bird-158', 'flowing-snowflake-159', 'avid-armadillo-160', 'radiant-fog-161'}
none_padding={'fiery-vortex-139', 'copper-star-140', 'spring-energy-141', 'firm-water-142', 'cosmic-dragon-143', 'lunar-frost-145', 'icy-cosmos-144', 'woven-moon-138'}




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
    info = pd.DataFrame(columns=['ID', 'model_type', 'batch_norm', 'batch_size',
                                 'loss_multiplication_factor', 'best_val_recall'])
    for run in runs:
        # check if the model was saved during training. If not, skip this run. It can be either because of the error
        # or because it is in the middle of training
        ID = run.name
        if ID not in average_padding and ID not in max_padding and ID not in none_padding:
            continue
        model_type = run.config['MODEL_TYPE']
        batch_norm = run.config['BATCH_NORM']
        batch_size = run.config['BATCH_SIZE']
        padding_mode = 'average' if ID in average_padding else 'max' if ID in max_padding else 'none'
        loss_multiplication_factor = run.config['loss_multiplication_factor']
        best_val_recall = run.config['best_val_recall']
        info = pd.concat([info,
                          pd.DataFrame.from_dict(
                              {'ID': [ID], 'model_type': [model_type],
                               'padding_mode':[padding_mode],
                               'batch_norm': [batch_norm],
                               'batch_size': [batch_size],
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
            # add batch dimension if it is not there
            if len(labels.shape) == 1:
                labels = np.expand_dims(labels, axis=0)

            # softmax, transformation to numpy, argmax for each classification task
            outputs = [torch.softmax(output, dim=-1) for output in outputs]
            outputs = [output.cpu().numpy().squeeze() for output in outputs]
            outputs = [np.argmax(output, axis=-1, keepdims=True) for output in outputs]
            outputs = [output.squeeze() if output.shape != (1,) else output for output in outputs]
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
    info = info.reset_index(drop=True)
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
        batch_size = int(info['batch_size'].iloc[i])
        batch_norm = info['batch_norm'].iloc[i]
        padding = True if info['padding_mode'].iloc[i] != 'none' else False
        padding_mode = info['padding_mode'].iloc[i]
        transformer_layers = 1 if str(1) in model_type else 2
        # construct data loaders
        train_generator, dev_generator, test_generator = create_data_generators(data=data, batch_size=batch_size,
                                                                            padding=padding, padding_mode=padding_mode)
        # construct model
        model = Seq2one_model_unwindowed(input_size=256, num_classes=[3,3,3], transformer_num_heads=16,
                                     num_transformer_layers=transformer_layers, batch_norm=batch_norm)
        # load model weights
        model.load_state_dict(torch.load(os.path.join(output_path, info['ID'].iloc[i]+'.pth')))
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
        info.to_csv('/nfs/home/ddresvya/scripts/BEA/models/info_unwindowed.csv', index=False)

        # clear memory
        del model, train_generator, dev_generator, test_generator
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

