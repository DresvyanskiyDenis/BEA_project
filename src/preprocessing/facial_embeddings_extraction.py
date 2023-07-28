import sys
import os

sys.path.append('/work/home/dsu/BEA_project/')
sys.path.append('/work/home/dsu/datatools/')

from functools import partial

import pandas as pd
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm

from feature_extraction.face_recognition_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def load_embeddings_extractor(path_to_weights: str, device: torch.device) -> torch.nn.Module:
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=3,
                                     num_regression_neurons=None)
    model.load_state_dict(torch.load(path_to_weights))
    # cut off the last layer responsible for classification
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    print('The embeddings extractor is loaded.')
    return model


def extract_embeddings_df(input_df: pd.DataFrame, face_detector, preprocessing_functions, embeddings_extractor,
                          embeddings_size: int,
                          device: torch.device, output_folder: str, output_filename: str) -> pd.DataFrame:
    # create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # create embeddings_df
    column_names = ['video_name', 'frame_num', 'q1', 'q2', 'q3'] + ['emb_{}'.format(i) for i in range(embeddings_size)]
    embeddings_df = pd.DataFrame(columns=column_names)
    # save empty dataframe to create a file
    embeddings_df.to_csv(os.path.join(output_folder, output_filename), index=False)
    # initialize last embeddings
    last_embeddings = np.zeros((1, 256))
    # iterate over rows of input_df
    for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
        # load image
        img = np.array(Image.open(row['file_name']))
        # preprocess image (face recognition + cropping + ImageNet preprocessing)
        bbox = recognize_one_face_bbox(img, face_detector)
        if bbox is None:
            embeddings = last_embeddings
        else:
            # crop image to face
            face = extract_face_according_bbox(img, bbox)
            # transform np array to tensor
            face = torch.from_numpy(face)
            # change the order of dimensions as torch expects channels first
            face = face.permute(2, 0, 1)
            # preprocess image
            for preprocessing_function in preprocessing_functions:
                face = preprocessing_function(face)
            # add batch dimension
            face = face.unsqueeze(0)
            face = face.to(device)
            # extract embeddings
            embeddings = embeddings_extractor(face)
            embeddings = embeddings.detach().cpu().numpy()
            last_embeddings = embeddings
        # save embeddings
        new_row = {
            'video_name': row['video_name'],
            'frame_num': row['frame_num'],
            'q1': row['q1'],
            'q2': row['q2'],
            'q3': row['q3'],
        }
        new_row.update({'emb_{}'.format(i): embeddings[0][i] for i in range(embeddings_size)})
        embeddings_df = pd.concat([embeddings_df, pd.DataFrame.from_records([new_row])], ignore_index=True)
        # dump embeddings_df to disk every 1000 rows to save the CPU time as the pandas.concat copies every time the
        # whole dataframe and slows down the process
        if index % 1000 == 0:
            # write to disk
            embeddings_df.to_csv(os.path.join(output_folder, output_filename), index=False, mode='a', header=False)
            del embeddings_df
            # clear embeddings_df
            embeddings_df = pd.DataFrame(columns = column_names)

    # save remaining rows
    embeddings_df.to_csv(os.path.join(output_folder, output_filename), index=False, mode='a', header=False)


def main():
    # params
    data_path = r'/Data/'
    output_folder = r'/work/home/dsu/Datasets/BEA/extracted_embeddings/'
    path_to_train_labels = os.path.join(data_path, 'train.csv')
    path_to_dev_labels = os.path.join(data_path, 'dev.csv')
    path_to_test_labels = os.path.join(data_path, 'test.csv')
    path_to_train_images = os.path.join(data_path, 'train')
    path_to_dev_images = os.path.join(data_path, 'dev')
    path_to_test_images = os.path.join(data_path, 'test')
    batch_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_to_weights_embeddings_extractor = "/work/home/dsu/tmp/deep-capybara-42.pth"
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                               EfficientNet_image_preprocessor()]
    # load face detector
    face_detector = load_and_prepare_detector_retinaFace_mobileNet('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('The face detector is loaded.')
    # load embeddings extractor
    embeddings_extractor = load_embeddings_extractor(path_to_weights_embeddings_extractor, device)
    # load labels
    train_labels = pd.read_csv(path_to_train_labels)
    #dev_labels = pd.read_csv(path_to_dev_labels)
    #test_labels = pd.read_csv(path_to_test_labels)
    # change path to images to new path where these images are located
    train_labels['file_name'] = train_labels['file_name'].apply(lambda x: os.path.join(path_to_train_images, x))
    #dev_labels['file_name'] = dev_labels['file_name'].apply(lambda x: os.path.join(path_to_dev_images, x))
    #test_labels['file_name'] = test_labels['file_name'].apply(lambda x: os.path.join(path_to_test_images, x))

    # extract embeddings
    print('Extracting embeddings for train set...')
    extract_embeddings_df(input_df=train_labels, face_detector=face_detector, preprocessing_functions=preprocessing_functions,
                          embeddings_extractor=embeddings_extractor, embeddings_size=256,
                          device=device, output_folder=output_folder, output_filename='train_embeddings.csv')
    #print('Extracting embeddings for dev set...')
    #extract_embeddings_df(input_df=dev_labels, face_detector=face_detector, preprocessing_functions=preprocessing_functions,
    #                        embeddings_extractor=embeddings_extractor, embeddings_size=256,
    #                        device=device, output_folder=output_folder, output_filename='dev_embeddings.csv')
    #print('Extracting embeddings for test set...')
    #extract_embeddings_df(input_df=test_labels, face_detector=face_detector,
    #                      preprocessing_functions=preprocessing_functions,
    #                      embeddings_extractor=embeddings_extractor, embeddings_size=256,
    #                      device=device, output_folder=output_folder, output_filename='test_embeddings.csv')


if __name__ == "__main__":
    main()
