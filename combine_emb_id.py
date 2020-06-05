import numpy as np


def combine_id_emb(id_path, emb_npy_path, output):

    osmids = []
    combined_embeddings = []

    with open(id_path) as f:
        for line in f:
            id_ = line.strip().split(' ')
            osmids.append(id_[0])

    embeddings = np.load(emb_npy_path)

    assert len(osmids) == len(embeddings)

    for i in range(len(osmids)):
        _id = osmids[i]
        _emb = []
        _emb.append(_id)
        for ele in embeddings[i].tolist():
            _emb.append(str(ele))
        combined_embeddings.append(_emb)

    with open(output, 'w+') as f:
        for emb in combined_embeddings:
            f.write(' '.join(emb))
            f.write('\n')


if __name__ == '__main__':

    id_txt_path = 'sanfrancisco/train_data/label_is_turning_circle/sf_node_turning_circle_val_embeddings.txt'
    emb_npy_path = 'sanfrancisco/train_data/label_is_turning_circle/sf_node_turning_circle_val_embeddings.npy'
    combined_emb_path = 'sanfrancisco/train_data/label_is_turning_circle/sf_graphSAGE_node_turning_circle.embeddings'

    combine_id_emb(id_txt_path, emb_npy_path, combined_emb_path)