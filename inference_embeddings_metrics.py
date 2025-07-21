import pandas as pd
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.data import load_dataset_and_get_model_name
from sklearn.decomposition import PCA


def make_key(row):
    return f"{row['function_name']}|{row['compiler']}|{row['version']}|{row['opt']}|{row['bin']}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, 'inference/datasets')
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, 'inference/embeddings')
    os.makedirs(output_dir, exist_ok=True)

    # get all dataset files
    all_datasets = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('-embeddings')]

    # load dataset
    model_names = []
    teacher_dataset = None
    student_datasets = []

    for dataset_path in all_datasets:
        dataset, model_name = load_dataset_and_get_model_name(dataset_path)

        if model_name == 'clap':
            teacher_dataset = dataset
        else:
            model_names.append(model_name)
            student_datasets.append(dataset)
        

    print('loaded all datasets')
    # build key mapper
    teacher_dict = {make_key(row): row["embedding"] for row in tqdm(teacher_dataset, desc='build key mapper...')}

    # use PCA to reduce from teacher -> student
    all_embeddings = np.array(list(teacher_dict.values()))
    pca = PCA(n_components=128)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # build reduced key mapper
    keys = list(teacher_dict.keys())
    reduced_teacher_dict = {k: reduced_embeddings[i] for i, k in tqdm(enumerate(keys), desc='build reduced key mapper', total=len(teacher_dict))}

    # metrics for each student
    metrics = defaultdict(list)

    for model_name, student_dataset in tqdm(zip(model_names, student_datasets), desc='calculating metrics for each student...', total=len(model_names)):
        results = []

        for row in student_dataset:
            key = make_key(row)
            if key in reduced_teacher_dict:

                if model_name == 'distil_projected':
                    teacher_embedding = np.array(teacher_dict[key])
                    student_embedding = np.array(row['embedding'])

                    metrics[f'Cosine_{model_name}'].append(np.dot(row["embedding"], teacher_dict[key]))
                    metrics[f'MSE_{model_name}'].append(mean_squared_error(student_embedding, teacher_embedding))
                    metrics[f'MAE_{model_name}'].append(mean_absolute_error(student_embedding, teacher_embedding))
                
                else:
                    teacher_embedding = np.array(reduced_teacher_dict[key])
                    student_embedding = np.array(row['embedding'])

                    metrics[f'Cosine_{model_name}'].append(np.dot(row["embedding"], reduced_teacher_dict[key]))
                    metrics[f'MSE_{model_name}'].append(mean_squared_error(student_embedding, teacher_embedding))
                    metrics[f'MAE_{model_name}'].append(mean_absolute_error(student_embedding, teacher_embedding))

   
    # avg the metrics
    final_metrics = defaultdict(dict)

    for key, values in metrics.items():
        avg = np.nanmean(values)

        parts = key.split("_")
        model = parts[-1]
        metric_name = "_".join(parts[:-1]) 

        if metric_name.endswith("_ranking"):
            metric_name = metric_name.replace("_ranking", "")

        if metric_name.endswith("_distil"):
            metric_name = metric_name.replace("_distil", "")

        print(f'model: {model}, {metric_name}: {avg:.4f}') 
        final_metrics[model][metric_name] = avg

    # save data
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # save plot
    metrics_to_plot = [
        'Cosine',
        'MAE',
        'MSE',
    ]

    # fixed color for models
    model_colors = {
        'hard': 'tab:blue',
        'random': 'tab:cyan',
        'distil': 'tab:orange',
        'baseline': 'tab:green',
        'clap': 'tab:red',
    }

    # create a plot for each of the desired metrics
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(8, 5)) 
        
        models_for_plot = []
        values_for_plot = []
        colors_for_plot = []

        # Collect data for the current metric across all models
        for model, model_data in final_metrics.items():
            if metric_name in model_data:
                models_for_plot.append(model)
                values_for_plot.append(model_data[metric_name])
                colors_for_plot.append(model_colors.get(model, 'gray'))


        # sort models alphabetically
        sorted_indices = np.argsort(models_for_plot)
        models_for_plot = [models_for_plot[i] for i in sorted_indices]
        values_for_plot = [values_for_plot[i] for i in sorted_indices]
        colors_for_plot = [colors_for_plot[i] for i in sorted_indices]

        bars = plt.bar(models_for_plot, values_for_plot, color=colors_for_plot, edgecolor='black')
        
        plt.title(f'{metric_name} Across Models') # More descriptive title
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha='right') # Rotate and align x-axis labels
        plt.grid(True, axis='y', linestyle='--', alpha=0.7) # Add grid for readability

        # Annotate bars with values
        for bar, val in zip(bars, values_for_plot):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='darkred') # Added color

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_barplot.png"))
        plt.close()

    print(f"Generated {len(metrics_to_plot)} bar plots")