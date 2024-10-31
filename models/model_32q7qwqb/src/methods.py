import dropbox
import json
import os
import sys
import glob
import copy
import shutil
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.insert(0, f"{os.getenv('MODEL_DIR')}/src")
from dataset import *
from architecture import *

def set_device(verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if verbose:
            print("Using GPU!")
    else:
        if verbose:
            print("Could not find GPU! Using CPU only.")
        if os.getenv('ENVIRONMENT') == "colab":
            if verbose:
                print("If you want to enable GPU, go to Runtime > View Resources > Change Runtime Type and select GPU.")
    return device

def upload_model_weights(model, dbx_access_token, delete=True):
    if dbx_access_token == "":
        print("Dropbox access token uninitialized. Unable to upload model weights.")
        return
    try:
        dbx = dropbox.Dropbox(dbx_access_token)
    except:
        print("Could not connect to Dropbox when attempting to upload model weights.")
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

def initialize_model(device, dbx_access_token, lookback, reset_weights=False):
    model = scene_perception_model(lookback=lookback).to(device)
    if reset_weights:
        print("Resetting model weights.")
        return model
    if dbx_access_token == "":
        raise ValueError("Dropbox access token uninitialized. Could not download model weights. Using default model weights.")
    try:
        model = download_model_weights(model, dbx_access_token)
    except:
        raise ValueError("Could not download model weights. Using default model weights.")
    return model

def create_datasets(lookback, device=None, datasets=None, include_all_datasets=True, include_unity_datasets=False, 
                    include_real_world_datasets=False, val_ratio=.2):

    if device is None:
        device = set_device()

    dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"

    # Get data dirs from user specified datasets
    if datasets is not None:
        train_val_data = []
        for dataset in datasets:
            dataset_data_dirs = glob.glob(f"{dataset_dir}/{dataset}/data/*")
            dataset_data_idxs = [int(os.path.splitext(os.path.basename(dir))[0]) for dir in dataset_data_dirs]
            train_val_data.extend([{'dataset': dataset, 'idx': idx} for idx in dataset_data_idxs])
        if len(train_val_data) == 0:
            print("No datasets found. Check that the dataset names are correct.")
            sys.exit()

    # Get data dirs from sample
    elif not (include_all_datasets or include_unity_datasets or include_real_world_datasets):
        sample_dataset = "sample/sample_dataset"
        dataset_data_dirs = glob.glob(f"{dataset_dir}/{sample_dataset}/data/*")
        dataset_data_idxs = [int(os.path.splitext(os.path.basename(dir))[0]) for dir in dataset_data_dirs]
        train_val_data = [{'dataset': dataset, 'idx': idx} for idx in dataset_data_idxs]

    # Get data dirs from all datasets
    else:
        train_val_data = []
        for dataset_category in ["unity", "real_world"]:
            # Check to skip the category if not requested
            if not include_all_datasets and (
                (dataset_category == "unity" and not include_unity_datasets) or
                (dataset_category == "real_world" and not include_real_world_datasets) ):
                continue
            category_data_dirs = glob.glob(f'{dataset_dir}/{dataset_category}/*/data/*')
            category_data_idxs = [int(os.path.splitext(os.path.basename(dir))[0]) for dir in category_data_dirs]
            category_dataset_names = [dir.replace('\\','/').split(f"{dataset_category}/")[1].split("/data/")[0] for dir in category_data_dirs]
            train_val_data.extend([{'dataset': f"{dataset_category}/{dataset_name}", 'idx': idx} for idx, dataset_name in zip(category_data_idxs, category_dataset_names)])

    # Split into train and val and create datasets
    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=random.randint(1, 100))
    train_dataset = Dataset_Class(data=train_data, augment=True, device=device, label_input_threshold=.1, lookback=lookback)
    val_dataset = Dataset_Class(data=val_data, augment=False, device=device, label_input_threshold=.1, lookback=lookback)

    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=32, val_batch_size=100):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    return train_dataloader, val_dataloader

def get_performance_metrics(conf_matrix, num_classes=4):
    epsilon = 1e-8
    class_names = ["Background", "Lane Lines", "Drivable Area", "Cones"]
    metrics = {}

    # Accuracy
    total_pixels = conf_matrix.sum()
    accuracy = conf_matrix.trace() / total_pixels
    metrics['Accuracy'] = accuracy

    # Per-class metrics
    for class_idx in range(num_classes):
        TP = conf_matrix[class_idx, class_idx]
        FN = conf_matrix[class_idx, :].sum() - TP
        FP = conf_matrix[:, class_idx].sum() - TP
        TN = total_pixels - TP - FN - FP

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        specificity = TN / (TN + FP + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = TP / (TP + FP + FN + epsilon)
        dice_coefficient = 2 * TP / (2 * TP + FP + FN + epsilon)

        metrics[f'{class_names[class_idx]} Precision'] = precision
        metrics[f'{class_names[class_idx]} Recall'] = recall
        metrics[f'{class_names[class_idx]} Specificity'] = specificity
        metrics[f'{class_names[class_idx]} F1 Score'] = f1_score
        metrics[f'{class_names[class_idx]} IoU'] = iou
        metrics[f'{class_names[class_idx]} Dice Coefficient'] = dice_coefficient

    # Mean metrics
    mean_iou = sum(metrics[f'{class_names[class_idx]} IoU'] for class_idx in range(num_classes)) / num_classes
    mean_dice_coefficient = sum(metrics[f'{class_names[class_idx]} Dice Coefficient'] for class_idx in range(num_classes)) / num_classes
    metrics['Mean IoU'] = mean_iou
    metrics['Mean Dice Coefficient'] = mean_dice_coefficient

    return metrics

def train_model(model, criterion, optimizer, train_dataloader):
    model.train()
    _, data, label = next(iter(train_dataloader))
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, torch.argmax(label, dim=3))
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_model(model, dataloader, num_classes=4):
    with torch.no_grad():
        model.eval()
        _, data, label = next(iter(dataloader))
        output = model(data)
        B, C, W, H = output.shape
        output = output.reshape(B * W * H, C)
        label = label.reshape(B * W * H, C)
        output_binary = output.argmax(dim=1).cpu()
        label_binary = label.argmax(dim=1).cpu()
        conf_matrix = confusion_matrix(label_binary, output_binary, labels=list(range(num_classes)))
        metrics = get_performance_metrics(conf_matrix, num_classes)
        return metrics

def training_loop(model, criterion, optimizer, train_dataloader, val_dataloader, dbx_access_token, val_dataset,
                  num_epochs=50, critiqueing_metric="Accuracy", upload_weights_to_dropbox=True,
                  auto_stop=False, auto_stop_patience=10,
                  verbose=True, log_every_n_epochs=5, log_no_improvement_every_n_epochs=10,
                  display_sample_results=False, num_sample_results=2, display_sample_results_every_n_epochs=20):

    train_loss_hist = []
    val_performance_hist = []
    epochs_since_best_val_performance = 0

    if display_sample_results:
        print("Initial sample results:")
        show_sample_results(model, val_dataset, set_device(verbose=False), num_samples=num_sample_results)
    
    print("Starting training...")

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

        if (epoch == 1 or epoch % log_every_n_epochs == 0 or epoch == num_epochs) and verbose:
            print(f"[EPOCH {epoch}/{num_epochs}]  Train Loss: {train_loss:.4f}  <>  Val Accuracy: {100*val_performance['Accuracy']:.2f}%  <>  Val Mean IoU: {100*val_performance['Mean IoU']:.2f}%")

        if epochs_since_best_val_performance > 0 and epochs_since_best_val_performance % log_no_improvement_every_n_epochs == 0 and verbose:
            print(f"[EPOCH {epoch}/{num_epochs}]  No improvement in validation {critiqueing_metric} for {epochs_since_best_val_performance} epochs")

        if display_sample_results and epoch % display_sample_results_every_n_epochs == 0:
            show_sample_results(model, val_dataset, set_device(verbose=False), num_samples=num_sample_results)

        if auto_stop and epochs_since_best_val_performance >= auto_stop_patience:
            print(f"Training auto stopped. No improvement in validation accuracy for {auto_stop_patience} epochs.")
            break

    print(f"\nTraining done!\n{epoch} epochs completed")
    print(f"Final Model Metrics:  Train Loss: {train_loss:.4f}  <>  Val Accuracy: {100*val_performance['Accuracy']:.2f}%  <>  Val Mean IoU: {100*val_performance['Mean IoU']:.2f}%\n")

    model.load_state_dict(best_model_state_dict)

    if not upload_weights_to_dropbox:
        print("Skipping uploading model weights to dropbox.\nIf you wish to upload them later, run 'upload_model_weights(model, dbx_access_token)'.")
    else:
        try:
            upload_model_weights(model, dbx_access_token)
        except Exception as e:
            print(f"Could not upload model weights to Dropbox - {e}")

    return model, train_loss_hist, val_performance_hist, best_val_performance

def test_model_on_benchmarks(model, device, lookback, all_benchmarks=False, benchmarks=[], report_results=True,
                             visualize_sample_results=True, num_sample_results=3, print_results=True):

    if not all_benchmarks and len(benchmarks) == 0:
        print("No benchmarks specified. Please specify benchmarks to test on.")
        print("Either set all_benchmarks=True or provide a list of benchmark names in the benchmarks argument.")
        return

    model.eval()

    dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"
    benchmark_dir = f"{dataset_dir}/benchmarks"

    # Get all benchmark names if requested
    if all_benchmarks:
        benchmark_data_dirs = glob.glob(f"{benchmark_dir}/*/data/*")
        benchmark_dataset_names = [dir.replace('\\','/').split(f"benchmarks/")[1].split("/data/")[0] for dir in benchmark_data_dirs]
        benchmarks = list(set(benchmark_dataset_names))

    # Loop through benchmarks
    for benchmark in benchmarks:
        
        # Create dataset for benchmark
        benchmark_data_dirs = glob.glob(f'{os.getenv("ROOT_DIR")}/datasets/benchmarks/{benchmark}/data/*')
        if len(benchmark_data_dirs) == 0:
            print(f"No data found for benchmark: \"{benchmark}.\"")
            continue
        benchmark_data_idxs = [int(os.path.splitext(os.path.basename(dir))[0]) for dir in benchmark_data_dirs]
        benchmark_data = [{'dataset': f"benchmarks/{benchmark}", 'idx': idx} for idx in benchmark_data_idxs]
        benchmark_dataset = Dataset_Class(data=benchmark_data, augment=False, device=device, lookback=lookback)

        # Get metrics for benchmark
        num_classes = 4
        conf_matrix_total = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for _, data, label in tqdm(benchmark_dataset, desc=f'Testing on {benchmark}', unit=' frame'):
                data = data.unsqueeze(0) # Add batch dimension
                output = model(data)
                B, C, W, H = output.shape
                output = output.reshape(B * W * H, C)
                label = label.reshape(B * W * H, C)
                output_binary = output.argmax(dim=1).cpu()
                label_binary = label.argmax(dim=1).cpu()
                conf_matrix = confusion_matrix(label_binary, output_binary, labels=list(range(num_classes)))
                conf_matrix_total += conf_matrix
        metrics = get_performance_metrics(conf_matrix_total, num_classes)

        if print_results:
            print(f'\n{benchmark} metrics:')
            for metric in metrics:
                print(f'\t{metric}: {metrics[metric]:.4f}')
            print()

        if report_results:
            model_performance_dir = f"{os.getenv('MODEL_DIR')}/content/performance.json"
            with open(model_performance_dir, 'r') as file:
                model_performance_json = json.load(file)
            model_performance_json[benchmark] = metrics
            with open(model_performance_dir, 'w') as file:
                json.dump(model_performance_json, file, indent=4)
            print(f"Metrics saved in performance.json for benchmark \"{benchmark}\".")

        if visualize_sample_results:
            show_sample_results(model, benchmark_dataset, device, num_samples=num_sample_results)

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

def show_sample_results(model, dataset, device, num_samples=4, occupancy_zoom_out_factor=0.48, 
                        occupancy_horizontal_stretch_factor=1.0, occupancy_vertical_stretch_factor=1.0):
    
    if num_samples < 2:
        num_samples = 2
    rand_indices = random.sample(range(len(dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 4, figsize=(12, 4*num_samples))
    for i, idx in enumerate(rand_indices):
        raw_data, data, label = dataset[idx]

        # Format and display data
        axs[i, 0].imshow(cv2.cvtColor(raw_data.detach().squeeze().permute(1,2,0).clamp(0,1).cpu().numpy(), cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title("Data")

        # Format and display label
        label = label.argmax(dim=2)
        label_display = torch.zeros(raw_data.shape, device=device)
        label_display[0][label == 0], label_display[1][label == 0], label_display[2][label == 0] = 190, 190, 190
        label_display[0][label == 1], label_display[1][label == 1], label_display[2][label == 1] = 220, 0, 0
        label_display[0][label == 2], label_display[1][label == 2], label_display[2][label == 2] = 155, 240, 160
        label_display[0][label == 3], label_display[1][label == 3], label_display[2][label == 3] = 230, 140, 0
        label_display = label_display.permute(1,2,0).byte().cpu().numpy()
        axs[i, 1].imshow(label_display)
        axs[i, 1].set_title("Label")

        # Format and display model output
        data = data.unsqueeze(0)
        model_output = model(data).squeeze().argmax(dim=0)
        output_display = torch.zeros(raw_data.shape, device=device)
        output_display[0][model_output == 0], output_display[1][model_output == 0], output_display[2][model_output == 0] = 190, 190, 190
        output_display[0][model_output == 1], output_display[1][model_output == 1], output_display[2][model_output == 1] = 220, 0, 0
        output_display[0][model_output == 2], output_display[1][model_output == 2], output_display[2][model_output == 2] = 155, 240, 160
        output_display[0][model_output == 3], output_display[1][model_output == 3], output_display[2][model_output == 3] = 230, 140, 0
        output_display = output_display.permute(1,2,0).byte().cpu().numpy()
        axs[i, 2].imshow(output_display)
        axs[i, 2].set_title("Model Output")

        # Format and display top down occupancy grid
        width, height = output_display.shape[1], output_display.shape[0]
        # Create the homography matrix
        pts1 = np.float32([
            ( width * .25 , height * 0 ), #tl
            ( width * 0   , height * 1 ), #bl
            ( width * .75 , height * 0 ), #tr
            ( width * 1   , height * 1 )  #br
        ])
        occupancy_zoom_out_factor_adjusted = max(occupancy_zoom_out_factor/2, .05)
        occupancy_zoom_out_factor_adjusted = min(occupancy_zoom_out_factor_adjusted, 0.45)
        lower_bound = 0.5 - occupancy_zoom_out_factor_adjusted
        upper_bound = 0.5 + occupancy_zoom_out_factor_adjusted
        dst_pts = np.float32([
            ( width * lower_bound , height * lower_bound), #tl
            ( width * lower_bound , height * upper_bound), #bl
            ( width * upper_bound , height * lower_bound), #tr
            ( width * upper_bound , height * upper_bound)  #br
        ])
        homography_matrix = cv2.getPerspectiveTransform(pts1, dst_pts)
        # Create occupancy grid
        occupancy_grid = cv2.warpPerspective(output_display, homography_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        # Remove all leading and trailing black rows and columns
        gray = cv2.cvtColor(occupancy_grid, cv2.COLOR_RGB2GRAY)
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        occupancy_grid = occupancy_grid[y:y+h, x:x+w]
        # Stretch occupancy grid for accurate physical length representation
        new_width = int(w * occupancy_horizontal_stretch_factor)
        new_height = int(h * occupancy_vertical_stretch_factor)
        occupancy_grid = cv2.resize(occupancy_grid, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        axs[i, 3].imshow(occupancy_grid)
        axs[i, 3].set_title("Model Output\nOccupancy Grid")
    
    plt.show()

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