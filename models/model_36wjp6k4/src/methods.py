import dropbox
import json
import os
import sys
import re
import glob
import copy
import shutil
import matplotlib.pyplot as plt
from getpass import getpass
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.insert(0, f"{os.getenv('MODEL_DIR')}/src")
from dataset import *
from architecture import *

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU!")
    else:
        print("Could not find GPU! Using CPU only.")
        if os.getenv('ENVIRONMENT') == "colab":
            print("If you want to enable GPU, go to Runtime > View Resources > Change Runtime Type and select GPU.")
    return device

def upload_model_weights(model, dbx_access_token, delete=True):
    if dbx_access_token == "":
        print("Dropbox access token uninitialized. Unable to upload model weights.")
        return
    try:
        dbx = dropbox.Dropbox(dbx_access_token)
    except:
        print("Could not connect to Dropbox when attempting to upload weights.")
        return
    dbx_model_weight_dir = f'/UMARV/ComputerVision/ScenePerception/model_weights/model_{os.getenv("MODEL_ID")}_weights.pth'
    local_model_weights_dir = f'{os.getenv("REPO_DIR")}/models/model_{os.getenv("MODEL_ID")}/content/weights.pth'   
    torch.save(model.state_dict(), local_model_weights_dir)
    with open(local_model_weights_dir, 'rb') as file:
        dbx.files_upload(file.read(), dbx_model_weight_dir, mode=dropbox.files.WriteMode("overwrite"))
    print("Uploaded model weights to Dropbox.")
    if delete:
        os.remove(local_model_weights_dir)

def download_model_weights(model, dbx_access_token, delete=True):
    if dbx_access_token == "":
        print("Dropbox access token uninitialized. Unable to download model weights.")
        return
    try:
        dbx = dropbox.Dropbox(dbx_access_token)
    except:
        print("Could not connect to Dropbox when attempting to download weights. Using default weights.")
        return model
    dbx_model_weight_dir = f'/UMARV/ComputerVision/ScenePerception/model_weights/model_{os.getenv("MODEL_ID")}_weights.pth'
    local_model_weights_dir = f'{os.getenv("REPO_DIR")}/models/model_{os.getenv("MODEL_ID")}/content/weights.pth'   
    try:
        metadata = dbx.files_get_metadata(dbx_model_weight_dir)
    except dropbox.exceptions.ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            print("No model weights found in Dropbox.")
            return model
    file_metadata, res = dbx.files_download(dbx_model_weight_dir)
    with open(local_model_weights_dir, 'wb') as file:
        file.write(res.content)
    model.load_state_dict(torch.load(local_model_weights_dir))
    print("Downloaded model weights from Dropbox.")
    if delete:
        os.remove(local_model_weights_dir)
    return model

def initialize_model(device, dbx_access_token, reset_weights=False):
    model = lane_model().to(device)
    if reset_weights:
        return model
    if dbx_access_token == "":
        print("Dropbox access token uninitialized. Using default weights.")
    try:
        model = download_model_weights(model, dbx_access_token)
    except:
        print("Could not download model weights. Using default weights.")
    return model

def create_datasets(device=None, datasets=None, benchmarks=None, include_all_datasets=True,
                    include_unity_datasets=False, include_real_world_datasets=False, val_ratio=.2):

    if device is None:
        device = set_device()

    dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"

    # Get data dirs from user specified datasets
    if datasets is not None:
        all_train_val_data_dirs = []
        for dataset in datasets:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            all_train_val_data_dirs.extend(dataset_data_dirs)
        if len(all_train_val_data_dirs) == 0:
            print("No datasets found. Check that the dataset names are correct.")
            sys.exit()

    else:

        # Get data dirs from sample
        if not (include_all_datasets or include_unity_datasets or include_real_world_datasets):
            sample_dataset_dir = "sample/sample_dataset"
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{sample_dataset_dir}/data/*")
            all_train_val_data_dirs.extend(dataset_data_dirs)

        # Get data dirs from all datasets
        else:
            all_train_val_data_dirs = []
            for dataset_category in ["unity", "real_world"]:
                # Check to skip the category if not requested
                if not include_all_datasets and (
                    (dataset_category == "unity" and not include_unity_datasets) or
                    (dataset_category == "real_world" and not include_real_world_datasets) ):
                    continue
                category_data_dirs = glob.glob(f'{dataset_dir}/{dataset_category}/*/data/*')
                all_train_val_data_dirs.extend(category_data_dirs)

    # Get train/val label directories
    all_train_val_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in all_train_val_data_dirs]

    # Split into train and val
    train_data_dirs, val_data_dirs, train_label_dirs, val_label_dirs = train_test_split(
        all_train_val_data_dirs, all_train_val_label_dirs, test_size=val_ratio, random_state=random.randint(1, 100)
    )

    # Get data dirs from user specified benchmarks
    if benchmarks is not None:
        all_benchmark_data_dirs = []
        for dataset in benchmarks:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            all_benchmark_data_dirs.extend(dataset_data_dirs)
    
    # Get data dirs from all benchmarks
    else:
        all_benchmark_data_dirs = glob.glob(f'{dataset_dir}/benchmarks/*/data/*')

    # Get benchmark label directories
    all_benchmark_label_dirs = [re.sub(r'\bdata\b', 'label', data_dir) for data_dir in all_benchmark_data_dirs]

    # Create train/val/benchmark datasets
    train_dataset = Dataset_Class(
        data_dirs=train_data_dirs, label_dirs=train_label_dirs, device=device, label_input_threshold=.1
    )
    val_dataset = Dataset_Class(
        data_dirs=val_data_dirs, label_dirs=val_label_dirs, device=device, label_input_threshold=.1
    )
    benchmark_dataset = Dataset_Class(
        data_dirs=all_benchmark_data_dirs, label_dirs=all_benchmark_label_dirs, device=device, label_input_threshold=.1
    )

    return train_dataset, val_dataset, benchmark_dataset

def create_dataloaders(train_dataset, val_dataset, benchmark_dataset, batch_size=32, val_size=100):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)
    benchmark_dataloader = DataLoader(benchmark_dataset, batch_size=50, shuffle=False)
    return train_dataloader, val_dataloader, benchmark_dataloader

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

def train_model(model, criterion, optimizer, train_dataloader):
    model.train()
    _, data, label = next(iter(train_dataloader))
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_model(model, dataloader):
    with torch.no_grad():
        model.eval()
        _, data, label = next(iter(dataloader))
        output = model(data)
        B, C, W, H = output.shape
        output = output.reshape(B * W * H, C)
        label = label.reshape(B * W * H, C)
        output_binary = output.argmax(dim=1).cpu()
        label_binary = label.argmax(dim=1).cpu()
        conf_matrix = confusion_matrix(label_binary, output_binary)
        TN_total = conf_matrix[0, 0]
        FP_total = conf_matrix[0, 1]
        FN_total = conf_matrix[1, 0]
        TP_total = conf_matrix[1, 1]
        metrics = get_performance_metrics(TN_total, FP_total, FN_total, TP_total)
        return metrics

def training_loop(model, criterion, optimizer, train_dataloader, val_dataloader, dbx_access_token, num_epochs=50, critiqueing_metric="Mean Pixel Accuracy", auto_stop=True, auto_stop_patience=10):

    train_loss_hist = []
    val_performance_hist = []
    epochs_since_best_val_performance = 0

    for epoch in tqdm(range(1, num_epochs+1), desc='Training', unit='epoch'):
        train_loss = train_model(model, criterion, optimizer, train_dataloader)
        train_loss_hist.append(train_loss)
        val_performance = validate_model(model, val_dataloader)
        val_performance_hist.append(val_performance)

        if epoch == 1 or val_performance[critiqueing_metric] > best_val_performance[critiqueing_metric]:
            best_model_state_dict = model.state_dict()
            best_val_performance = copy.deepcopy(val_performance)
            epochs_since_best_val_performance = 0
        else:
            epochs_since_best_val_performance += 1
        
        if auto_stop and epochs_since_best_val_performance >= auto_stop_patience:
            print(f'Epoch: {epoch}/{num_epochs}  <>  Train Loss: {train_loss:.4f}  <>  Val MPA: {100*val_performance["Mean Pixel Accuracy"]:.2f}%  <>  Val Precision: {100*val_performance["Precision"]:.2f}%')
            print(f'Training auto stopped. No improvement in validation accuracy for {auto_stop_patience} epochs.')
            break

        if (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            print(f'Epoch: {epoch}/{num_epochs}  <>  Train Loss: {train_loss:.4f}  <>  Val MPA: {100*val_performance["Mean Pixel Accuracy"]:.2f}%  <>  Val Precision: {100*val_performance["Precision"]:.2f}%')

    model.load_state_dict(best_model_state_dict)
    upload_model_weights(model, dbx_access_token)

    return model, train_loss_hist, val_performance_hist, best_val_performance

def test_model_on_benchmarks(model, device, all_benchmarks=True, benchmarks=None, report_results=True,
                             visualize_sample_results=True, print_results=True):
    
    model.eval()

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
        benchmark_dataset = Dataset_Class(data_dirs=benchmark_data_dirs, label_dirs=benchmark_label_dirs,
                                          device=device, label_input_threshold=.1)

        with torch.no_grad():
            TN_total, FP_total, FN_total, TP_total = 0, 0, 0, 0
            for _, data, label in tqdm(benchmark_dataset, desc=f'Testing on {benchmark}', unit=' frame'):
                data = data.unsqueeze(0) # Add batch dimension
                output = model(data)
                B, C, W, H = output.shape
                output = output.reshape(B * W * H, C)
                label = label.reshape(B * W * H, C)
                output_binary = output.argmax(dim=1).cpu()
                label_binary = label.argmax(dim=1).cpu()
                conf_matrix = confusion_matrix(label_binary, output_binary)
                TN_total += conf_matrix[0, 0]
                FP_total += conf_matrix[0, 1]
                FN_total += conf_matrix[1, 0]
                TP_total += conf_matrix[1, 1]
            
        metrics = get_performance_metrics(TN_total, FP_total, FN_total, TP_total)

        if report_results:
            model_performance_dir = f"{os.getenv('MODEL_DIR')}/content/performance.json"
            with open(model_performance_dir, 'r') as file:
                model_performance_json = json.load(file)
            model_performance_json[benchmark] = metrics
            with open(model_performance_dir, 'w') as file:
                json.dump(model_performance_json, file, indent=4)
            print("Metrics saved in performance.json.")

        if print_results:
            print(f'\n{benchmark} metrics:')
            for metric in metrics:
                print(f'\t{metric}: {metrics[metric]:.4f}')
            print()

        if visualize_sample_results:
            show_sample_results(model, benchmark_dataset, device, num_samples=5)

    return

def graph_loss_history(loss_hist, split=''):
    plt.figure()
    plt.plot(torch.tensor(loss_hist, device='cpu'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{split} Loss History')
    plt.show()

def graph_performance_history(performance_hist, split='', metrics=['Accuracy']):
    for metric in metrics:
        plt.figure()
        plt.plot(torch.tensor([performance[metric] for performance in performance_hist], device='cpu'))
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{split} {metric} History')
        plt.show()

def show_sample_results(model, dataset, device, output_threshold=.5, num_samples=5):
    rand_indices = random.sample(range(len(dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 4, figsize=(12, 4*num_samples))
    for i, idx in enumerate(rand_indices):
        raw_data, data, label = dataset[idx]
        data = data.unsqueeze(0)
        model_output = model(data)
        soft = nn.Softmax(dim=1)
        soft_output = soft(model_output)
        soft_ones_output = soft_output[0,1,:,:]
        ones_output = torch.zeros(soft_ones_output.shape, device=device)
        ones_output[soft_ones_output > output_threshold] = 1
        soft_ones_output[0,0], soft_ones_output[0,1] = 1, 0 # Prevent imshow from normalizing the image
        axs[i, 0].imshow(cv2.cvtColor(raw_data.detach().squeeze().permute(1,2,0).clamp(0,1).cpu().numpy(), cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title("Data")
        axs[i, 1].imshow(label.detach().permute(1,2,0)[:,:,1].squeeze().clamp(0,1).cpu().numpy(), cmap='gray')
        axs[i, 1].set_title("Label")
        axs[i, 2].imshow(soft_ones_output.detach().squeeze().clamp(0,1).cpu().numpy(), cmap='gray')
        axs[i, 2].set_title("Soft Output")
        axs[i, 3].imshow(ones_output.detach().squeeze().detach().clamp(0,1).cpu().numpy(), cmap='gray')
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