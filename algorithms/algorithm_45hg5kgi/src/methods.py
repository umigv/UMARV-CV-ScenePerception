import dropbox
import json
import os
import sys
import re
import glob
import copy
import matplotlib.pyplot as plt
from getpass import getpass
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.insert(0, f"{os.getenv('ALGORITHM_DIR')}/src")
from dataset import *
from lane_detector import *

def create_dataset(datasets=None, include_all_datasets=True,
                   include_unity_datasets=False, include_real_world_datasets=False):

    dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"

    data_dirs = []

    # Get data dirs from user specified datasets
    if datasets is not None:
        for dataset in datasets:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            data_dirs.extend(dataset_data_dirs)
        if len(data_dirs) == 0:
            print("No datasets found. Check that the dataset names are correct.")
            sys.exit()

    else:

        # Get data dirs from sample
        if not (include_all_datasets or include_unity_datasets or include_real_world_datasets):
            sample_dataset_dir = "sample/sample_dataset"
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{sample_dataset_dir}/data/*")
            data_dirs.extend(dataset_data_dirs)

        # Get data dirs from all datasets
        else:
            for dataset_category in ["unity", "real_world"]:
                # Check to skip the category if not requested
                if not include_all_datasets and (
                    (dataset_category == "unity" and not include_unity_datasets) or
                    (dataset_category == "real_world" and not include_real_world_datasets) ):
                    continue
                category_data_dirs = glob.glob(f'{dataset_dir}/{dataset_category}/*/data/*')
                data_dirs.extend(category_data_dirs)

    label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in data_dirs]

    dataset = Dataset_Class(
        data_dirs=data_dirs, label_dirs=label_dirs, label_input_threshold=25
    )
  
    return dataset

def get_performance_metrics(TN_total, FP_total, FN_total, TP_total):
    epsilon = 1e-8
    tn_rate = TN_total / (TN_total + FP_total + epsilon)
    fp_rate = FP_total / (TN_total + FP_total + epsilon)
    tp_rate = TP_total / (TP_total + FN_total + epsilon)
    fn_rate = FN_total / (TP_total + FN_total + epsilon)
    accuracy = (TN_total + TP_total) / (TN_total + FP_total + FN_total + TP_total)
    precision = TP_total / (TP_total + FP_total + epsilon)
    recall = TP_total / (TP_total + FN_total + epsilon)
    specificity = TN_total / (TN_total + FP_total + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    iou_lane = TP_total / (TP_total + FP_total + FN_total + epsilon)
    iou_background = TN_total / (TN_total + FP_total + FN_total + epsilon)
    m_iou = (iou_lane + iou_background) / 2
    total_pixels = TN_total + FP_total + FN_total + TP_total
    pixel_accuracy = (TN_total + TP_total) / (total_pixels + epsilon)
    mean_pixel_accuracy = (iou_lane + iou_background) / 2
    class_frequency = [TP_total + FN_total, TN_total + FP_total]
    fw_iou = (iou_lane * class_frequency[0] + iou_background * class_frequency[1]) / (total_pixels + epsilon)
    dice_coefficient = 2 * TP_total / (2 * TP_total + FP_total + FN_total + epsilon)
    boundary_f1_score = 2 * TP_total / (2 * TP_total + FP_total + FN_total + epsilon)
    metrics = {
        'TN Rate': tn_rate,
        'FP Rate': fp_rate,
        'TP Rate': tp_rate,
        'FN Rate': fn_rate,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1_score,
        'IoU Lane': iou_lane,
        'IoU Background': iou_background,
        'Mean IoU': m_iou,
        'Pixel Accuracy': pixel_accuracy,
        'Mean Pixel Accuracy': mean_pixel_accuracy,
        'Frequency-Weighted IoU': fw_iou,
        'Dice Coefficient': dice_coefficient,
        'Boundary F1 Score': boundary_f1_score
    }
    return metrics

def test_algorithm_on_benchmarks(all_benchmarks=True, benchmarks=None, report_results=True,
                                 visualize_sample_results=True, print_results=True):

    if all_benchmarks:
        all_benchmark_data_dirs = glob.glob(f'{os.getenv("ROOT_DIR")}/datasets/benchmarks/*/data/*.jpg')
        benchmarks = []
        for benchmark_data_dir in all_benchmark_data_dirs:
            benchmark = re.search(r'benchmark_\w+', benchmark_data_dir).group()
            if benchmark not in benchmarks:
                benchmarks.append(benchmark)
    
    for benchmark in benchmarks:
        
        benchmark_data_dirs = glob.glob(f'{os.getenv("ROOT_DIR")}/datasets/benchmarks/{benchmark}/data/*')
        benchmark_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in benchmark_data_dirs]
        benchmark_dataset = Dataset_Class(data_dirs=benchmark_data_dirs, label_dirs=benchmark_label_dirs)

        TN_total, FP_total, FN_total, TP_total = 0, 0, 0, 0
        for data, label in tqdm(benchmark_dataset, desc=f"Testing on {benchmark} "):
            output = lane_detector(data)
            output_binary = (output > .5).astype(np.uint8).flatten()
            label_binary = label[:,:,1].flatten()
            conf_matrix = confusion_matrix(label_binary, output_binary)
            if conf_matrix.shape != (1, 1):
                TN_total += conf_matrix[0, 0]
                FP_total += conf_matrix[0, 1]
                FN_total += conf_matrix[1, 0]
                TP_total += conf_matrix[1, 1]
            # Handle 1x1 confusion matrix because sklearn is weird
            # Happens when output perfectly matches label
            elif conf_matrix.shape == (1, 1):
                if output_binary[0] == label_binary[0]:
                    TP_total += conf_matrix[0, 0]
                else:
                    TN_total += conf_matrix[0, 0]
        metrics = get_performance_metrics(TN_total, FP_total, FN_total, TP_total)

        if report_results:
            algorithm_performance_dir = f"{os.getenv('ALGORITHM_DIR')}/content/performance.json"
            with open(algorithm_performance_dir, 'r') as file:
                algorithm_performance_json = json.load(file)
            algorithm_performance_json[benchmark] = metrics
            with open(algorithm_performance_dir, 'w') as file:
                json.dump(algorithm_performance_json, file, indent=4)
            print("Results successfully reported.")

        if print_results:
            print(f'\n{benchmark} metrics:')
            for metric in metrics:
                print(f'\t{metric}: {metrics[metric]:.4f}')

        if visualize_sample_results:
            show_sample_results(benchmark_dataset, num_samples=3)

    return

def show_sample_results(dataset, num_samples=5):
    rand_indices = random.sample(range(len(dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 4, figsize=(15, 2.5*num_samples))
    for i, idx in enumerate(rand_indices):
        data, label = dataset[idx]
        output = lane_detector(data)
        binary_output = (output > .5).astype(np.uint8)
        output[0,0], output[0,1] = 1, 0
        axs[i, 0].imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title("Data")
        axs[i, 1].imshow(label[:,:,1], cmap='gray')
        axs[i, 1].set_title("Label")
        axs[i, 2].imshow(output, cmap='gray')
        axs[i, 2].set_title("Output")
        axs[i, 3].imshow(binary_output, cmap='gray')
        axs[i, 3].set_title("Binary Output")

def upload_datasets_to_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    source_directory = '/content/datasets/'
    destination_directory = '/content/drive/My Drive/UMARV/ScenePerception/datasets/'
    shutil.copytree(source_directory, destination_directory)

def get_datasets_from_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    source_directory = '/content/drive/My Drive/UMARV/ScenePerception/datasets/'
    destination_directory = '/content/datasets/'
    shutil.copytree(source_directory, destination_directory)