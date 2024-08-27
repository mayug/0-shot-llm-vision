# Do Vision and Language Encoders Represent the World Similarly?

This project is an implementation of the paper "Do Vision and Language Encoders Represent the World Similarly?" accepted CVPR 2024. 


This project investigates the alignment between vision and language encoders, specifically analyzing whether unimodal vision and language models, which represent the same physical world, share semantic similarities in their latent spaces. We find that well trained vision and language encoders have high semantic similarity between their encoder spaces. Using Centered Kernel Alignment (CKA) and seeded graph-matching techniques, we further explore the potential for matching unaligned encoders without additional training. The research introduces two methods: Fast Quadratic Assignment Problem optimization and a novel localized CKA-based matching, demonstrating their effectiveness across tasks like cross-lingual, cross-domain caption matching, and image classification. The findings suggest that vision encoders trained on large datasets exhibit semantic similarities to language encoders, enabling zero-shot communication between these modalities.  




![asd](/images/combined_v10_with_star.png)
*Kernel CKA and QAP Matching accuracy are correlated with the training set size and quality of the training set. Here
the language encoder is kept constant to the best BERT-sentence encoder (i.e.All-Roberta-large-v1). There is a clear correlation between
CKA and QAP Matching accuracy across all architectures, training paradigm and data regimes*



![asd](/images/cka_max_simple_illustration_v5_flip.png)
*For matching, we calculate the kernels for image and
text embeddings and employ QAP-based seeded matching to max-
imize CKA for obtaining the optimal permutation P . For retrieval,
we append query embeddings to base embeddings and retrieve the
best caption that maximizes the local CKA for a query image*


## Setup 
To set up the project environment using conda, follow these steps: 

1. Clone the repository: ```git clone https://github.com/mayug/0-shot-llm-vision.git```
2. Navigate to the project directory: ```cd 0-shot-llm-vision```
3. Create a new conda environment: ```conda create --name vlm python=3.10```
4. Install appropiate pytorch version from [here](https://pytorch.org/get-started/locally/).
5. Install the required dependencies: ```pip install -r requirements.txt```

Now you have a conda environment set up with all the necessary dependencies installed.

## Folder Structure

The project has the following folder structure:    

- `data`: This folder stores the retrieved text and image embeddings. The file naming convention is `{dataset}_{model_name}_img.pt` for image embeddings anv
- `results`: This folder stores the algorithm results averaged over the considered seeds.
- `seedwise`: This folder stores the algorithm results for each individual seed.
- `src`: This folder contains the code for the algorithms used in the project.
  - `utils.py`: This file contains utility functions used by other scripts.
- `cluster.py`: This script is used for clustering the embeddings (required later in the pipeline).
- `get_embeds.py`: This script is responsible for retrieving the text and image embeddings.
- `run.py`: This script runs the main algorithm of the project.

## Code Explanation

## Usage

To use this project:

1. Ensure that the required dependencies are installed.
2. Specify COCO and NOCAPS dataset location in `get_embeds.py` file.
3. Run `get_embeds.py` to retrieve the text and image embeddings for the desired models and datasets.
4. Run `cluster.py` to perform clustering on the embeddings if needed.
5. Run `run.py` with the desired command-line arguments to execute the main algorithm and generate the results.

The results will be saved in the `results` and `seedwise` folders as CSV files.

Please refer to the individual script files for more detailed information on their usage and functionalities.

### `get_embeds.py`

The `get_embeds.py` script is responsible for retrieving text and image embeddings from various models. It supports the following models:

- ConvNeXt
- DINO-V2
- CLIP
- all-roberta
- DETR (with different backbones and components)
- SAM
- SegFormer
- DPT
- ResNet-101
- ViT

The script takes the following command-line arguments:
- `--m` or `--model_name`: Specifies the model name (default: "dinov2").
- `--d` or `--dataset`: Specifies the dataset (default: "coco").
- `--gpu`: Specifies the GPU to use (default: 0).

The script loads the specified model and dataset, and then runs the model on the dataset to extract the embeddings. The extracted embeddings are saved as PyTorch tensors in the `data` folder.

Example command:
```python get_embeds.py --m dinov2 --d coco --gpu 0```

### `cluster.py`

The `cluster.py` script performs clustering on the text or image embeddings. It uses the K-means clustering algorithm to cluster the embeddings into a specified number of clusters. The script then finds the closest data points to the cluster centers based on cosine similarity and saves their indices to a file in the `data` folder.

The script takes the following command-line arguments:
- `--m` or `--model_name`: Specifies the model name (default: "dinov2").
- `--d` or `--dataset`: Specifies the dataset (default: "coco").
- `-t` or `--text`: Enables text clustering (action flag).
- `--nc` or `--n_clusters`: Specifies the number of clusters (default: 320).

Example command:
```python cluster.py --m dinov2 --d coco --nc 320```

### `run.py`

The `run.py` script is the main entry point of the project. It performs the following steps:

1. Parses command-line arguments to configure the experiment. The available flags are:
   - `--m` or `--method`: Specifies the baseline method to use (default: "linear").
   - `--b` or `--base_samples`: Specifies the number of base samples (default: 320).
   - `--q` or `--query_samples`: Specifies the number of query samples (default: 500).
   - `--c` or `--clustering_mode`: Specifies the clustering mode (default: 0).
   - `--vid_model`: Specifies the vision model to use (default: "dinov2").
   - `--text_model`: Specifies the text model to use (default: "allroberta").
   - `--base_data`: Specifies the base dataset (default: "coco").
   - `--query_data`: Specifies the query dataset (default: "nocaps").
   - `-str` or `--stretch`: Enables stretching (action flag).
   - `-vtl` or `--vtl`: Enables vision-to-language (action flag).
   - `--gpu`: Specifies the GPU to use (default: 0).

Example command:
```python run.py --m linear --b 320 --q 500 --c 0 --vid_model dinov2 --text_model allroberta --base_data coco --query_data nocaps --gpu 0```


If you find our work useful, please cite us. 

```bibtex
@inproceedings{maniparambil2024vision,
  title={Do Vision and Language Encoders Represent the World Similarly?},
  author={Maniparambil, Mayug and Akshulakov, Raiymbek and Djilali, Yasser Abdelaziz Dahou and El Amine Seddik, Mohamed and Narayan, Sanath and Mangalam, Karttikeya and O'Connor, Noel E},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14334--14343},
  year={2024}
}
