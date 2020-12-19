from os import name
import pandas as pd
import networkx as nx
import dgl
if __name__ == "__main__":
    # pandas reads csv
    edges_data = pd.read_csv('../data/knowledge_aquisition_reference.csv')
    # networkx reads pandas
    g_nx: nx.DiGraph = nx.from_pandas_edgelist(edges_data,
                                               'paper_id',
                                               'reference_id',
                                               create_using=nx.DiGraph())

    # dgl read networkx
    # ATTENTION!!!: nodes in dgl graph is ordered by paperid
    g = dgl.from_networkx(g_nx)
