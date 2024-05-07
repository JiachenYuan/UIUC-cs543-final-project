# Image Classification

## How to Run the Code?
1. Replicate the project directory:
```
/{workspace}
    /data
    /checkpoint
    /logs
    /losses_and_accuracies
    {rest of files...}
```

2. Run `chmod +x data_prep.sh && ./data_prep.sh` to download the extract the Tiny ImageNet dataset. You might need to put all the test images from `tiny-imagenet-200/test/images` to a new folder under `tiny-imagenet-200/test/images/test`

3. Each individual IPython Notebook corresponds to an experiment. Each notebook is named in the format: `[model]_[dataset].ipynb`