"""
DirectLiNGAM-based causal feature ordering script.

This script uses the DirectLiNGAM algorithm to discover the causal ordering of
numerical geochemical variables from a tabular CSV dataset. Non-feature columns,
such as sample IDs, coordinates, and labels, are removed before model fitting.
The resulting causal order is saved as `causal_feature_order_custom.npy`, which
can be used by subsequent model training and prediction scripts to reorder
geochemical feature channels.

The original CSV file used to generate the causal order is confidential and
cannot be provided. Instead, the generated result file
`causal_feature_order_custom.npy` is provided for reproducibility and downstream
experiments.

Usage:
1. If access to the confidential CSV data is available, place the CSV file in
   the working directory and name it `confidential_not_available.csv`.
2. Run this script with Python.
3. The script will fit a DirectLiNGAM model and save the discovered causal
   feature order to `causal_feature_order_custom.npy`.

Main output:
- `causal_feature_order_custom.npy`
"""

import pandas as pd
import networkx as nx
from lingam import DirectLiNGAM
import numpy as np


def discover_causal_graph(df):
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.drop(
        columns=["FID", "POINT_X", "POINT_Y", "XX", "YY", "Label", "label", "OBJECTID"],
        errors="ignore"
    )

    model = DirectLiNGAM()
    model.fit(numeric_df.values)

    causal_graph = nx.DiGraph()
    feature_names = list(numeric_df.columns)

    causal_graph.add_nodes_from(feature_names)

    for i, var in enumerate(feature_names):
        for j, weight in enumerate(model.adjacency_matrix_[i, :]):
            if weight != 0:
                causal_graph.add_edge(feature_names[j], feature_names[i], weight=float(weight))

    return causal_graph, model, feature_names


if __name__ == "__main__":
    input_csv = "confidential_not_available.csv"

    df = pd.read_csv(input_csv)

    causal_graph, model, all_features = discover_causal_graph(df)

    causal_order = [all_features[i] for i in model.causal_order_]

    np.save("causal_feature_order_custom.npy", np.array(causal_order, dtype=object))

    print("=" * 80)
    print("DirectLiNGAM Causal Order")
    print("=" * 80)
    print(" -> ".join(causal_order))

    print("=" * 80)
    print("Saved causal order to causal_feature_order_custom.npy")
    print("=" * 80)