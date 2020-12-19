from operator import itemgetter
import pandas as pd
import networkx as nx


def read_edge(file_name: str):
    return pd.read_csv(file_name)


def print_list(l: list, filename: str) -> None:
    with open(filename, 'w') as f:
        f.write("\"paper_id\",\"reference_count\"\n")
        for item in l:
            f.write('\"' + str(item[0]) + '\",\"' + str(item[1]) + '\"\n')


if __name__ == "__main__":
    # pandas reads csv
    edges_data = read_edge('../data/knowledge_aquisition_reference.csv')

    # networkx reads pandas
    g_nx: nx.DiGraph = nx.from_pandas_edgelist(edges_data,
                                               'paper_id',
                                               'reference_id',
                                               create_using=nx.DiGraph())

    nd_list = []
    for n in g_nx.nodes:
        nd_list.append((n, g_nx.in_degree(n)))
    sort_by_id = sorted(nd_list, key=itemgetter(0))
    sort_by_ref = sorted(nd_list, key=itemgetter(1), reverse=True)
    # print_list(sort_by_id, '../data/ref_count_id.csv')
    # print_list(sort_by_ref, '../data/ref_count_ref.csv')

    ranked = []
    for i, r in sort_by_id:
        if r < 10:
            ranked.append((i, 0))
            pass
        elif r < 30:
            ranked.append((i, 1))
        elif r < 100:
            ranked.append((i, 2))
        elif r < 300:
            ranked.append((i, 3))
        elif r < 1000:
            ranked.append((i, 4))
        else:
            ranked.append((i, 5))

    print_list(ranked,'../data/rank_id.csv')
