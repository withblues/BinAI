<h1 style="text-align:center;">Binary Analysis with AI SoSe 2025 </h1>

<h4 align="center">
<p>
<a href=#about>About</a> |
<a href=#quickstart>QuickStart</a> |
<a href=#details>Details</a> 
</h4>

## About
With the rising perfomance of pre-trained models on various domains, this project tries to leverage these models on binary analysis. Instead of using already promising models like CLAP [Wang et al. (2024)](https://arxiv.org/abs/2402.16928) to distil the knoledge to smaller models. This project explores different distillation approaches with a pre-trained BERT model and focuses on the Downstream Task Binary Code Similarity Detection

## QuickStart
This document will help you reproduce the results from the distillation.

### Requirements
 - Python 3.10 or higher
 - PyTorch
 - Transformers library
 - Ideally a CUDA-enabled GPU
 - dataset

 For easy replication use the following command in a conda environment:

 ```bash
conda env create -f environment.yml
```
### Dataset & BERT model

The dataset is accecisble under this [Github Repository](https://github.com/Cisco-Talos/binary_function_similarity). The BERT model is pretrained with a MLM task on that dataset.

### Precomputing Data

Since the dataset is huge and we use the data multiple times, we want to precompute the embeddings with CLAP for the different splits train/valid/test.
 ```bash
python cla_precompute.py --data_dir outputs/baseline-train.pkl --output_dir outputs/clap-train.pkl --indexing --split train/valid/test
```

In order to be able to experiment with the training setup we also pre tokenize our data for the BERT model. Otherwise, we would always need to tokenize for each run

 ```bash
python pretokenize_data.py --data_dir outputs --output_dir outputs --split train/val
```

All the data are either saved as pkl file or as a huggingface Datasets

### Training

This project follows two different training approaches. Both are accesible via the same script train.py
The first approach is to compare the embeddings of CLAP with BERT
 ```bash
python train.py --data_dir outputs --output_dir outputs --mode distil 
```

The second approach uses CLAP as soft labels. The teacher model gives different pair-wise cosine similarity scores. Therefore we need a strategy to sample anchors and corresponding target functions. I implemented two distinct sampling approaches. The random based sampling approach can be precomputed with:
 ```bash
python similarity_random_precompute.py --data_dir outputs --output_dir outputs --split train/val --top_k 10
```
The FAISS based sampling, which tries to include more true positive and true negatives examples can be executed like this:
 ```bash
python similarity_hard_mined_precompute.py --data_dir outputs --output_dir outputs --split train/val --top_k 10
```

Afterwards we can train the model with:
 ```bash
python train.py --data_dir outputs --output_dir outputs --mode ranking --function_pool hard_mined/random
```

### Inference

After training a diversity of models we wan't to evaluate them on on Binary Code Similarity Detection Task. Once again we want to precompute the data with our new models. Therfore we need to execute the following script.
 ```bash
python inference_precompute.py --data_dir outputs --output_dir outputs --model clap/distil/distil_projected/ranking_random/ranking_hard --batch_size 32
```

Afterwards we can use the embeddings to calculate the cosine similarity score for the function pool with:
 ```bash
python inference_ranking_similarity.py --data_dir outputs --output_dir outputs --model ranking_hard
 --model clap/distil/distil_projected/ranking_random/ranking_hard 
```

To now evaluate on common metrics like NCDG@k / MRR / Precision@k we use this command which will calculate all metrics for all the trained models if their corresponding data is precomputed
 ```bash
python inference_ranking_metrics.py --data_dir outputs --output_dir outputs --top_k 10
```


To also evaluate the embeddings of the each model to the teacher CLAP use this:
 ```bash
python inference_embeddings_metrics.py --data_dir outputs --output_dir outputs
```

## Details
The results of the distillation process can be seen in the report.

## License
This repository's source code is available under the [Apache-2.0 License](LICENSE).