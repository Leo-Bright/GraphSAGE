import json
import numpy as np


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


# generate idx for every node from node_2_embeddings dict
def gen_node_idx_json(node2emb, output_json_path):

    node2idx = {}

    count = 0
    for key in node2emb:
        node2idx[key] = count
        count += 1

    with open(output_json_path, 'w+') as f:
        json.dump(node2idx, f)


def gen_node_emb_npy(node2emb, node2idx, output_feats_path):

    embs = list(range(len(node2emb)))

    for node in node2emb:
        emb = node2emb[node]
        embs[node2idx[node]] = emb

    embeddings = np.vstack(embs)
    np.save(output_feats_path, embeddings)


# generate class_map for every node from node_2_embeddings dict
def gen_node_class_json(node2emb, node2tag, output_json_path):

    node2class = {}

    for node in node2emb:
        label = [1, 0] if node in node2tag else [0, 1]
        node2class[node] = label

    with open(output_json_path, 'w+') as f:
        json.dump(node2class, f)


if __name__ == '__main__':

    node_emb_dict = trans_input_file_to_ndarray('sanfrancisco/embeddings/sanfrancisco_raw_feature_none.embeddings')

    # step 0-1: generate the init id_map.json
    # node_idx_json_path = 'sanfrancisco/train_data/sanfrancisco-id_map.json'
    # gen_node_idx_json(node_emb_dict, node_idx_json_path)

    # step 0-2: generate feats.npy file
    # node_feats_npy_path = 'sanfrancisco/train_data/sanfrancisco-feats.npy'
    # with open('sanfrancisco/train_data/sanfrancisco-id_map.json') as f:
    #     node_idx_dict = json.loads(f.readline())
    # gen_node_emb_npy(node_emb_dict, node_idx_dict, node_feats_npy_path)

    # feats = np.load('sanfrancisco/train_data/sanfrancisco-feats.npy')

    # step 0-3: generate class_map.json file
    # node_tag_dict = json.load(open('sanfrancisco/osm_data/nodes_turning_circle.json'))
    # node_class_json_path = 'sanfrancisco/train_data/sanfrancisco-class_map.json'
    # gen_node_class_json(node_emb_dict, node_tag_dict, node_class_json_path)

    node2class = json.load(open('sanfrancisco/train_data/sanfrancisco-class_map.json'))

    print('done')