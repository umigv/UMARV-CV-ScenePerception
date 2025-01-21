# Working With Cloud Environments

## Google Colab

Google Colab is a great environment to work out of as it has free GPU access. These GPUs are limited and run out, so try to get your code working locally first, then hop on google colab to leverage the GPUs during training.

Pre-requisite: **Have a model already created and pushed to the branch**. Tutorial here: [https://github.com/umigv/UMARV-CV-ScenePerception/blob/main/docs/creating_models.md](https://github.com/AwrodHaghiTabrizi/UMARV-CV-ScenePerception/blob/main/docs/creating_models.md).

1. Go to [https://github.com/umigv/UMARV-CV-ScenePerception/branches](https://github.com/AwrodHaghiTabrizi/UMARV-CV-ScenePerception/branches).
2. Select your branch.
3. Navigate to your model foder.
4. Navigate to the google colab notebook and click into it.
    - Ex: `models/model_abc123/src/notebooks/colab_env.ipynb`
5. Copy your current url link.
6. Go to [https://colab.research.google.com/](https://colab.research.google.com/).
7. Click "GitHub".
8. Paste the URL from step 5 for the colab notebook.
9. Click on the magnifying glass.
10. Click on the correct `.ipynb` file for your model (usually the first result).
11. If stuck on an empty page, try refreshing the page.

If you wan't to utilize the GPU, navigate to: Runtime > View Resources > Change Runtime Type and select GPU.

## LambdaLabs

...

## Jetson

...