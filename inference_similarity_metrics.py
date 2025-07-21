import pandas as pd
import os
import argparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.data import load_df_and_get_model_name
from src.utils.metrics import normalized_dcg, mean_reciprocal_rank, precision_at_k



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, 'inference/cosine_similarity')
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, 'inference/similarity')
    os.makedirs(output_dir, exist_ok=True)
    top_k = args.top_k

    # get all csv files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('-results-cosine.csv')]

    # merge all dfs into one
    merged_df = None
    model_names = []

    for csv_file in all_files:
        df, model_name = load_df_and_get_model_name(csv_file)
        model_names.append(model_name)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=[
                "anchor_function_bin", "anchor_function_name", "anchor_compiler", "anchor_version", "anchor_opt",
                "target_function_bin", "target_function_name", "target_compiler", "target_version", "target_opt"
            ], how='inner')

    print('merged dataframes')

    # get similarity_columns and keep student names
    sim_columns = [col for col in merged_df.columns if col.startswith("sim_")]
    model_names.remove('clap')

    # create hard labels
    merged_df['ground_truth'] = (merged_df['anchor_function_name'] == merged_df['target_function_name']).astype(int)

    # metrics for each student   
    metrics = defaultdict(list)

    # group by anchor
    group_keys = [
        "anchor_function_bin", "anchor_function_name", "anchor_compiler",
        "anchor_version", "anchor_opt"
    ]
    

    for _, group in tqdm(merged_df.groupby(group_keys), desc='calculating metrics ...'):
        teacher_scores = group['sim_clap'].values
        ground_truth = group['ground_truth'].values


        for sim_col in sim_columns:
            model = sim_col.replace('sim_', '')
            student_scores = group[sim_col].values

            ### rank based metrics
            # sort student
            sorted_indices = np.argsort(student_scores)[::-1]
            sorted_relevance = ground_truth[sorted_indices]

            # calculate metrics
            metrics[f'MRR_{model}'].append(mean_reciprocal_rank(sorted_relevance))
            metrics[f'NDCG@{top_k}_{model}'].append(normalized_dcg(ground_truth, student_scores, top_k))
            metrics[f'Precision@{top_k}_{model}'].append(precision_at_k(sorted_relevance, top_k))

            if sim_col == "sim_clap":
                continue

            ### score based metrics
            mse = mean_squared_error(teacher_scores, student_scores)
            mae = mean_absolute_error(teacher_scores, student_scores)
            pcc, _ = pearsonr(teacher_scores, student_scores)
            scc, _ = spearmanr(teacher_scores, student_scores)

            metrics[f"MSE_{model}"].append(mse)
            metrics[f"MAE_{model}"].append(mae)
            metrics[f"Pearson_{model}"].append(pcc)
            metrics[f"Spearman_{model}"].append(scc)

        #break

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
        'MRR',
        f'NDCG@{top_k}',
        f'Precision@{top_k}',
        'MAE',
        'MSE',
        'Pearson',
        'Spearman'
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

        # collect data for the current metric across all models
        for model, model_data in final_metrics.items():
            if metric_name in model_data:
                models_for_plot.append(model)
                values_for_plot.append(model_data[metric_name])
                colors_for_plot.append(model_colors.get(model, 'gray'))


        if 'clap' in models_for_plot:
            clap_index = models_for_plot.index('clap')
            clap_model = models_for_plot.pop(clap_index)
            clap_value = values_for_plot.pop(clap_index)
            clap_color = colors_for_plot.pop(clap_index)
        else:
            # Handle case if clap not found
            clap_model = None
            clap_value = None
            clap_color = None

        # Sort remaining models by values
        sorted_indices = np.argsort(values_for_plot)
        models_sorted = [models_for_plot[i] for i in sorted_indices]
        values_sorted = [values_for_plot[i] for i in sorted_indices]
        colors_sorted = [colors_for_plot[i] for i in sorted_indices]

        # Put 'clap' at front if it was found
        if clap_model is not None:
            models_for_plot = [clap_model] + models_sorted
            values_for_plot = [clap_value] + values_sorted
            colors_for_plot = [clap_color] + colors_sorted
        else:
            models_for_plot = models_sorted
            values_for_plot = values_sorted
            colors_for_plot = colors_sorted

        bars = plt.bar(models_for_plot, values_for_plot, color=colors_for_plot, edgecolor='black')
        
        plt.title(f'{metric_name} Across Models') # More descriptive title
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.margins(y=0.1)
        plt.xticks(rotation=45, ha='right', fontsize=14) # Rotate and align x-axis labels
        plt.grid(True, axis='y', linestyle='--', alpha=0.7) # Add grid for readability

        # Annotate bars with values
        for bar, val in zip(bars, values_for_plot):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=14, color='black') # Added color

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_barplot.png"))
        plt.close()

    print(f"Generated {len(metrics_to_plot)} bar plots")