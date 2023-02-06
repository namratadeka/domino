import sys
sys.path.append('../')
import os

import argparse
import numpy as np
import pandas as pd
import meerkat as mk
from tqdm import tqdm

import torch
import dcbench

from domino.utils import seed_everything
from domino import embed, DominoSlicer
from dcbench import SliceDiscoverySolution
from domino.eval.metrics import compute_solution_metrics


def get_slices(tasks, slice_type='rare'):
    ids = []
    print(f'Collecting {slice_type} slices:')
    for id in tqdm(tasks):
        if tasks[id]._attributes['slice_category'] == slice_type:
            ids.append(id)
    return ids

def _run_domino(problem, n_slices, n_mixtures, lamda, emb_dp):
    print(problem)
    test_dp = mk.merge(problem["test_slices"], problem["test_predictions"], on="id")
    test_dp = mk.merge(problem["base_dataset"], test_dp, on="id")
    test_dp["pred"] = test_dp["probs"][:, 1]
    val_dp = mk.merge(problem["val_predictions"], problem["base_dataset"], on="id") 
    val_dp["pred"] = val_dp["probs"][:, 1]
    
    # Embed images
    # print('Embedding validation images:')
    # val_dp = embed(
    #     val_dp, 
    #     input_col="image",
    #     device=0
    # )
    # print('Embedding test images:')
    # test_dp = embed(
    #     test_dp, 
    #     input_col="image",
    #     device=0
    # )
    val_dp = val_dp.merge(emb_dp["id", "clip(image)"], on="id", how="left")
    test_dp = test_dp.merge(emb_dp["id", "clip(image)"], on="id", how="left")
    # Init. Domino
    domino_config = {
        'y_log_likelihood_weight':lamda,
        'y_hat_log_likelihood_weight':lamda,
        'n_mixture_components':n_mixtures,
        'n_slices': n_slices,
        'embedder': None
    }
    domino = DominoSlicer(
        **domino_config
    )
    # Fit on val-data
    print('Fitting on validation images:')
    domino.fit(data=val_dp, embeddings="clip(image)", targets="target", pred_probs="pred")

    # Predict on test-data
    print('Predicting slices for test images:')
    result = mk.DataPanel({"id": test_dp["id"]})
    result["slice_preds"] = domino.predict(
        test_dp, embeddings="clip(image)", targets="target", pred_probs="probs"
    )
    result["slice_probs"] = domino.predict_proba(
        test_dp, embeddings="clip(image)", targets="target", pred_probs="probs"
    )

    # Compute metrics
    print('Computing test metrics:')
    solution = SliceDiscoverySolution(
        artifacts={
            "pred_slices": result,
        },
        attributes={
            "problem_id": problem.id,
            "slicer_class": DominoSlicer,
            "slicer_config": domino_config,
            "embedding_column": 'clip(image)',
        },
    )
    metrics = compute_solution_metrics(
        solution,
    )

    for i in range(len(metrics)):
        metrics[i]['err_entropy'] = domino.err_entropy[metrics[i]['pred_slice_idx']]
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'domino evaluation script'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help='seed',
        default=0
    )
    parser.add_argument(
        "-s",
        "--slice",
        type=str,
        help='slice category from [rare, correlation, noisy].'
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        help='number of clusters'
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        help='number of mixture components'
    )
    parser.add_argument(
        "--lamda",
        type=float
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )

    (args, unknown_args) = parser.parse_known_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sd = dcbench.tasks["slice_discovery"]
    tasks = get_slices(sd.problems, slice_type=args.slice)
    outpath = f'./results/domino_nopca/{args.slice}/k{args.k}_m{args.m}_lambda_{args.lamda}_{args.seed}.csv'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    emb_dp = mk.DataPanel.read("./image_embs.mk")

    for task in tasks:
        try:
            result = _run_domino(sd.problems[task], args.k, args.m, args.lamda, emb_dp)
            result_df = pd.DataFrame(result)
            result_df.to_csv(outpath, mode='a', header=not os.path.exists(outpath))
        except:
            continue
