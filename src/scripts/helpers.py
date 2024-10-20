import os
import shutil
from datetime import datetime
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
from getpass import getpass
import dropbox
import http.client
from requests.exceptions import ChunkedEncodingError
import urllib3.exceptions
import requests

def copy_directory(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir, item)
        if os.path.isdir(source_item):
            copy_directory(source_item, destination_item)
        else:
            shutil.copy2(source_item, destination_item)

def base10_to_base36(number):
    if number == 0:
        return '0'
    base36_chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ''
    while number > 0:
        number, remainder = divmod(number, 36)
        result = base36_chars[remainder] + result
    return result

def copy_directory_from_dropbox(source_dir, destination_dir, dbx=None, dbx_access_token=None, use_thread=True):

    if dbx is None:
        if dbx_access_token is None:
            dbx_access_token = getpass.getpass("Enter your DropBox access token: ")
        dbx = dropbox.Dropbox(dbx_access_token)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    result = dbx.files_list_folder(source_dir)
    entries = result.entries
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)
    total_items = len(entries)

    def download_and_save(item, max_retries=5):
        source_item_path = item.path_display
        destination_item_path = os.path.join(destination_dir, os.path.basename(source_item_path))
        for retry in range(max_retries):
            try:
                if isinstance(item, dropbox.files.FolderMetadata):
                    copy_directory_from_dropbox(source_item_path, destination_item_path, dbx=dbx, use_thread=use_thread)
                else:
                    response = requests.get(dbx.files_get_temporary_link(source_item_path).link, timeout=60)
                    content = response.content
                    nparr = np.frombuffer(content, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    cv2.imwrite(destination_item_path, image)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"\nUnable to download: {item.path_display}\n\tError: {e}")
                
    if use_thread:
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(download_and_save, entries), total=total_items, desc=f"Downloading {source_dir} :", unit=" image"))
    else:
        for item in tqdm(entries, total=total_items, desc=f"Downloading {source_dir} :", unit=" image"):
            download_and_save(item)

def download_datasets_from_dropbox(dbx=None, dbx_access_token=None, use_thread=False, datasets=None, include_all_datasets=True):

    if dbx is None:
        if dbx_access_token is None:
            dbx_access_token = getpass("Enter your DropBox access token: ")
        dbx = dropbox.Dropbox(dbx_access_token)
    
    dbx_datasets_dir = '/UMARV/ComputerVision/ScenePerception/datasets'

    if datasets is not None:
        dataset_dirs = datasets
            
    else:
        dataset_dirs = []
        for dataset_category in ["real_world", "benchmarks"]:
            # Collect dataset image directories from DropBox
            dataset_category_dir = f"{dbx_datasets_dir}/{dataset_category}"
            result = dbx.files_list_folder(dataset_category_dir)
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FolderMetadata):
                    found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                    dataset_dirs.append(found_dataset_dir)
            while result.has_more:
                result = dbx.files_list_folder_continue(result.cursor)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FolderMetadata):
                        found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                        dataset_dirs.append(found_dataset_dir)

    for dataset_dir in dataset_dirs:
        copy_directory_from_dropbox(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"{os.getenv('ROOT_DIR')}/datasets/{dataset_dir}",
            dbx = dbx,
            dbx_access_token = dbx_access_token,
            use_thread = use_thread
        )