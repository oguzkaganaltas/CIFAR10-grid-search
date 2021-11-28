
# CIFAR10-grid-search

Oğuz Kağan Altaş 2385128


# How to run

## Dependencies

    conda create -n pytorch python=3.9
    conda activate pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install matplotlib
    pip install tqdm
	
## How to run the Script

on the pytorch environment we created
    `python assignment.py`

The best of each model will be saved into a directory called "best_models".
The results of the training will be saved into "RESULTS" file.