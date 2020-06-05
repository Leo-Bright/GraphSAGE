import json
import numpy as np
import random
from eval_scripts import graph
from graphsage import utils


# supports emb file and tag file
def trans_input_file_to_ndarray(input):

    support_suffix = ["emb", "embedding", "embeddings", "tag"]

    if input.rsplit('.',1 )[-1] not in support_suffix:
        raise BaseException("Only support emb and tag file.")

    output_ndarray_dic = {}

    convertion_float = lambda n: float(n)

    with open(input) as f:
        for line in f:
            node_others = line.strip().split(' ')
            node_id = node_others[0]
            others = node_others[1:]
            oth_array = [convertion_float(item) for item in others]
            if len(oth_array) > 0:
                output_ndarray_dic[node_id] = oth_array
    return output_ndarray_dic


# generate idx for every segment from seg_2_embeddings dict
def gen_seg_idx_json(seg2emb, output_json_path):

    seg2idx = {}

    for key in seg2emb:
        seg2idx[key] = int(key) - 1

    with open(output_json_path, 'w+') as f:
        json.dump(seg2idx, f)


def gen_seg_emb_npy(seg2emb, seg2idx, output_feats_path):

    embs = list(range(len(seg2emb)))

    for seg in seg2emb:
        emb = seg2emb[seg]
        embs[seg2idx[seg]] = emb

    embeddings = np.vstack(embs)
    np.save(output_feats_path, embeddings)


# generate class_map for every segment from segment_2_embeddings dict
def gen_seg_class_json(seg2emb, seg2tag, output_json_path):

    seg2class = {}

    for seg in seg2emb:
        label = [1, 0] if seg in seg2tag else [0, 1]
        seg2class[seg] = label

    with open(output_json_path, 'w+') as f:
        json.dump(seg2class, f)


def gen_graph_json(network, seg2idx, feats, seg2class, output_json_path):

    result_graph = {'directed': False, 'graph': {'name': 'disjoint_union( ,  )'}, 'nodes': [], 'links': [], 'multigraph': False}

    graph_size = len(seg2idx)
    test_size = graph_size // 10
    val_size = graph_size // 5

    links = []
    segs = [None for x in range(0, graph_size)]

    for src in seg2idx:

        src_idx = seg2idx[src]
        seg_feats = feats[src_idx].tolist()
        seg_class = seg2class[src]

        rd = random.randint(0, graph_size) + 1
        to_test = to_valid = False
        if rd < test_size:
            to_test = True
        elif rd < val_size:
            to_valid = True

        node = {'id': src, 'feature': seg_feats, 'label': seg_class, 'test': to_test, 'val': to_valid}
        try:
            segs[src_idx] = node
        except:
            print(src_idx)
            exit()

        targets = network[src]
        for targ in targets:
            targ_idx = seg2idx[targ]
            link = {'test_removed': False, 'train_removed': False, 'target': targ_idx, 'source': src_idx}
            links.append(link)

    result_graph['nodes'] = segs
    result_graph['links'] = links

    json.dump(result_graph, open(output_json_path, 'w+'))


if __name__ == '__main__':

    seg_emb_dict = trans_input_file_to_ndarray('sanfrancisco/embeddings/sanfrancisco_raw_feature_segment.embeddings')

    # step 0-1: generate the init id_map.json
    # seg_idx_json_path = 'sanfrancisco/train_data/sanfrancisco-segment-id_map.json'
    # gen_seg_idx_json(seg_emb_dict, seg_idx_json_path)

    # step 0-2: generate feats.npy file
    # seg_feats_npy_path = 'sanfrancisco/train_data/sanfrancisco-segment-feats.npy'
    # with open('sanfrancisco/train_data/sanfrancisco-segment-id_map.json') as f:
    #     seg_idx_dict = json.loads(f.readline())
    # gen_seg_emb_npy(seg_emb_dict, seg_idx_dict, seg_feats_npy_path)
    #
    # feats = np.load('sanfrancisco/train_data/sanfrancisco-segment-feats.npy')

    # step 0-3: generate class_map.json file
    # seg_tag_dict = json.load(open('sanfrancisco/osm_data/sf_segments_tiger_Blvd.json'))
    # seg_class_json_path = 'sanfrancisco/train_data/sanfrancisco-segment-class_map.json'
    # gen_seg_class_json(seg_emb_dict, seg_tag_dict, seg_class_json_path)

    # step 0-4: generate G.json file
    # network = graph.load_edgelist('sanfrancisco/osm_data/sanfrancisco_segment.network')['graph']
    # seg_graph_json_path = 'sanfrancisco/train_data/sanfrancisco-segment-G.json'
    # seg_idx_dict = json.load(open('sanfrancisco/train_data/sanfrancisco-segment-id_map.json'))
    # seg_feats_ndarray = np.load('sanfrancisco/train_data/sanfrancisco-segment-feats.npy')
    # seg_class_dict = json.load(open('sanfrancisco/train_data/sanfrancisco-segment-class_map.json'))
    # gen_graph_json(network, seg_idx_dict, seg_feats_ndarray, seg_class_dict, seg_graph_json_path)

    print('done')