# UMARV-CV-ScenePerception

A research and development platform for the University of Michigan's Autonomous Robotic Vehicle (UMARV) Computer Vision team to solve the understanding of a road scene.

## Models vs Algorithms

The models folder hosts all of our machine learning solutions, while the algorithms folder hosts our hard coded solutions. Each model/algorithm is seperated into its own folder and has its own unique ID.

## Scripts

The `src/scripts/` folder hosts our scripts which provide varrying functionalities from model/algorithm initialization, performance comparison, and dataset generation. To run them, right click on the script and select "Run Python File in Terminal".

## How To Interact With This Repository

[Video Tutorial](https://youtube.com) Coming Soon!!<!-- TODO Create video and add link -->

1. Have git installed on your computer.
    - [git installation guide](https://git-scm.com/downloads)
    - [git introduction](https://www.w3schools.com/git/git_intro.asp?remote=github)
2. Have Python installed on your computer.
    - [Python Installation Guide](https://wiki.python.org/moin/BeginnersGuide/Download)
3. Request access to the ScenePerception GitHub repository from a team lead.
    - You must accept the invitation to the GitHub repository.
4. Setup the repository on your local machine.
    - On your Desktop, right click and select 'Open In Terminal'.
    - ```mkdir UMARV```
    - ```cd UMARV```
    - ```mkdir ScenePerception```
    - ```cd ScenePerception```
    - ```git clone https://github.com/umigv/UMARV-CV-ScenePerception.git```
    - ```cd UMARV-CV-ScenePerception```
    - IMPORTANT: Replace your branch name in the end of the next 2 commands.
        - your_branch_name = "user/{your_name_with_no_spaces}"
        - Ex: Branch name for Awrod Haghi-Tabrizi = user/AwrodHaghiTabrizi
    - ```git checkout -b {your_branch_name}```
    - ```git push -u origin {your_branch_name}```
5. Open the project in VSCode.
    - Open VSCode.
    - Click File > Open Folder.
    - Open the `UMARV-CV-ScenePerception` folder.
        - Common mistake: Opening the `UMARV` folder or the `ScenePerception` folder.
        - IMPORTANT: Keep your working directory as `UMARV-CV-ScenePerception` when running scripts and notebooks.
6. Before starting development, install the following [Python libraries](https://github.com/umigv/UMARV-CV-ScenePerception/blob/main/docs/requirements.md) on your machine / virtual environment either manually or with ```pip install -r requirements.txt```.
    - ```pip install opencv-python```
    - ```pip install torch```
    - ```pip install scikit-learn```
    - ```pip install matplotlib```
    - ```pip install gitpython```
    - ```pip install dropbox```
    - ```pip install tqdm```
    - ```pip install nbformat```

### Repository Rules

- Full freedom to create/delete/edit code in your model/algorithm folder.
- Dont change any code in:
    - model/algorithm folders that dont belong to you (you can tell by the author name in the content/info.json or just by the model id itself).
    - `src/scripts/` (unless making global updates).
    - model_template/algorithm_tempalte (unless making global updates).
- Work in your own branch. Pull before every work session. Push after every work session.

## Environments

This repository allows development flexability to work in multiple environments, including:
    - Windows
    - Mac
    - Google Colab - [Working with Google Colab](https://github.com/umigv/UMARV-CV-ScenePerception/blob/main/docs/working_with_environments.md#google-colab)
    - LambdaLabs - [Working with LambdaLabs](https://github.com/umigv/UMARV-CV-ScenePerception/blob/main/docs/working_with_environments.md#lambdalabs)
    - Jetson (coming soon)

## Developing Models

1. Navigate to `src/scripts`.
2. Right click on either `create_model.py` or `create_copy_of_model.py`
    - `create_model.py` creates a new model from the template
    - `create_copy_of_model.py` creates a copy of a model using its model id
3. Click "Run Python File in Termainl".
4. Answer the prompts in the terminal.
5. Go through [Working With Models](https://github.com/umigv/UMARV-CV-ScenePerception/blob/main/docs/creating_models.md)

## Machine Learning Model Leaderboard

This leaderboard showcases the top performing segmentation models developed by UMARV members based on the average accuracy of the model during testing. Once you have developed a PyTorch model that has higher accuracy, you are free to add your model to the leaderboard by editing the main branch README.md. Still keep the model folder in your personal branch only.

| #   | Name | Accuracy | Other Metrics | Creators | Git Branch Name
| --- | ---- |---- | ---- | ---------- | ---- |
| 1   | 32mw3qk4 | 0.2390 | <details> Mean IoU: 0.1083; Mean Dice Coeffecient: 0.1872 </details> | Awrod | Main |

## Developing Algorithms

Functionality coming soon. For the time being, refer to the [LaneDetection repository](https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection) for Algorithms support.

<!--

1. Navigate to src/scripts
2. Right click on either "create_new_algorithm.py" or "create_copy_of_algorithm.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal
5. Go through [Working With Algorithms](https://github.com/AwrodHaghiTabrizi/UMARV-CV-ScenePerception/blob/users/AHT/docs/creating_algorithms.md)

-->
