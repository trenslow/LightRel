import ast
from parameters18 import *
import sys


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


def read_shape_file(file, unkwn):
    shps = {}
    with open(file) as f:
        for i, line in enumerate(f):
            shps[ast.literal_eval(line)] = i
    shps[unkwn] = len(shps)
    return shps


def read_feat_file(file, unkwn):
    feats = {}
    with open(file) as f:
        for i, item in enumerate(f):
            feats[item.strip()] = i
    feats[unkwn] = len(feats)
    return feats


def read_embeddings(file):
    embs = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                continue
            else:
                word, vect = ' '.join(split[:-300]), [float(val) for val in split[-300:]]  # -300 due to # of dims
                embs[word] = vect
    return embs


def read_clusters(file):
    clusts = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                wrd, clust = split
            else:
                continue
            if 'marlin' in file:
                clusts[wrd] = int(clust)
            elif 'brown' in file:
                clusts[wrd] = int(clust, 2)  # converts binary to int
    if 'brown' in file:
        clusts['<RARE>'] = max(clusts.values()) + 1
    return clusts


def pad_middle(sent, max_len):
    num_pads = max_len-len(sent)
    padding = num_pads * [None]
    if before_e2:
        return sent[:-1] + padding + [sent[-1]]
    else:
        return [sent[0]] + padding + sent[1:]


if __name__ == '__main__':
    which_set = sys.argv[1]
    if which_set == '0':
        which_set = ''
    which_record_train = 'record_train' + which_set + '.txt'
    which_record_test = 'record_test' + which_set + '.txt'
    records_and_outs = [(path_to_feat_folder + which_record_train, path_to_model_folder + 'libLinearInput_train.txt'),
                        (path_to_feat_folder + which_record_test, path_to_model_folder + 'libLinearInput_test.txt')]
    record_file = path_to_feat_folder + 'record.txt'
    vocab_file = path_to_feat_folder + 'vocab.txt'
    shapes_file = path_to_feat_folder + 'shapes.txt'
    label_file = path_to_feat_folder + 'labels.txt'
    e1_context_file = path_to_feat_folder + 'e1_context.txt'
    e2_context_file = path_to_feat_folder + 'e2_context.txt'
    abstracts_file = path_to_feat_folder + 'abstracts.txt'
    word_embds_file = path_to_feat_folder + 'abstracts-dblp-semeval2018.wcs.txt'  # smaller embds for dev
    # word_embds_file = path_to_feat_folder + 'acm_abstracts.wcs.txt'
    # cluster_file = path_to_feat_folder + 'dblp_marlin_clusters_1000'
    cluster_file = path_to_feat_folder + 'acm_marlin_clusters_1000'
    unknown = 'UNK'
    num_words = 0
    num_shapes = 0
    num_embeddings = 0
    num_clusters = 0
    num_e1_contexts = 0
    num_e2_contexts = 0
    num_abstracts = 0

    if fire_words:
        words = read_feat_file(vocab_file, unknown)
        num_words = len(words)
    if fire_clusters:
        clusters = read_clusters(cluster_file)
        num_clusters = max(clusters.values()) + 1
    if fire_shapes:
        shapes = read_shape_file(shapes_file, unknown)
        num_shapes = len(shapes)
    if fire_embeddings:
        embeddings = read_embeddings(word_embds_file)
        num_embeddings = len(list(embeddings.values())[0])
    if fire_e1_context:
        e1_contexts = read_feat_file(e1_context_file, unknown)
        num_e1_contexts = len(e1_contexts)
    if fire_e2_context:
        e2_contexts = read_feat_file(e2_context_file, unknown)
        num_e2_contexts = len(e2_contexts)

    with open(label_file) as labs:
        labels = {lab.strip(): i for i, lab in enumerate(labs, start=1)}

    len_token_vec = num_words + num_clusters + num_shapes + num_embeddings
    feat_val = ':1.0'

    for record_file, out_file in records_and_outs:
        records = read_record(record_file)
        if 'train' in record_file:
            max_rel_length = max([len(rec[1]) for rec in records])
            print('creating training file...')
        elif 'test' in record_file:
            print('creating test file...')

        with open(out_file, 'w+') as lib_out:
            for rec in records:
                sentence_feats = []
                current_relation = rec[4]
                sentence = rec[1]
                norm_sentence = pad_middle(sentence, max_rel_length)
                for i, token in enumerate(norm_sentence):
                    offset = i * len_token_vec
                    token_feats = []
                    if token:
                        if fire_words:
                            if token in words:
                                feat_pos = offset + words[token] + 1
                            else:
                                feat_pos = offset + words[unknown] + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_clusters:
                            if token in clusters:
                                feat_pos = offset + clusters[token] + num_words + 1
                            else:
                                feat_pos = offset + clusters['<RARE>'] + num_words + 1

                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_shapes:
                            shape_vec = [0, 0, 0, 0, 0, 0, 0]
                            if any(char.isupper() for char in token):
                                shape_vec[0] = 1
                            if ',' in token:
                                shape_vec[1] = 1
                            if any(char.isdigit() for char in token):
                                shape_vec[2] = 1
                            if i == 0 and token[0].isupper():
                                shape_vec[3] = 1
                            if token[0].islower():
                                shape_vec[4] = 1
                            if '_' in token:
                                shape_vec[5] = 1
                            if '"' in token:
                                shape_vec[6] = 1
                            tup_vec = tuple(shape_vec)
                            feat_pos = offset + shapes[tup_vec] + num_clusters + num_words + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_embeddings:
                            lowered = token.lower()
                            if lowered in embeddings:
                                vec = embeddings[lowered]
                                token_feats += [str(offset + n + num_shapes + num_clusters + num_words + 1) + ':' +
                                                str(vec[n]) for n in range(num_embeddings)]
                    else:
                        unknown_word = None
                        unknown_shape = None
                        unknown_cluster = None
                        # not handling unknown embeddings because they don't have indices
                        if fire_words:
                            feat_pos = offset + words[unknown] + 1
                            unknown_word = str(feat_pos) + feat_val
                        if fire_clusters:
                            feat_pos = offset + clusters['<RARE>'] + num_words + 1
                            unknown_cluster = str(feat_pos) + feat_val
                        if fire_shapes:
                            feat_pos = offset + shapes[unknown] + num_clusters + num_words + 1
                            unknown_shape = str(feat_pos) + feat_val
                        token_feats = [unknown_word, unknown_cluster, unknown_shape]

                    sentence_feats += token_feats
                sent_offset = len(norm_sentence) * len_token_vec
                if len(sentence) > 2:  # norm sentence not used to avoid padding elements
                    e1_context = sentence[1]
                    e2_context = sentence[-2]

                    if fire_e1_context and fire_e2_context:
                        e1_context_idx = e1_contexts[e1_context] if e1_context in e1_contexts else e1_contexts[unknown]
                        e2_context_idx = e2_contexts[e2_context] if e2_context in e2_contexts else e2_contexts[unknown]
                        e1_pos = sent_offset + e1_context_idx + 1
                        e2_pos = sent_offset + num_e1_contexts + e2_context_idx + 1
                        e1_feat = str(e1_pos) + feat_val
                        e2_feat = str(e2_pos) + feat_val
                        sentence_feats.append(e1_feat)
                        sentence_feats.append(e2_feat)
                    elif fire_e1_context:
                        e1_context_idx = e1_contexts[e1_context] if e1_context in e1_contexts else e1_contexts[unknown]
                        e1_pos = sent_offset + e1_context_idx + 1
                        e1_feat = str(e1_pos) + feat_val
                        sentence_feats.append(e1_feat)
                    elif fire_e2_context:
                        e2_context_idx = e2_contexts[e2_context] if e2_context in e2_contexts else e2_contexts[unknown]
                        e2_pos = sent_offset + e2_context_idx + 1
                        e2_feat = str(e2_pos) + feat_val
                        sentence_feats.append(e2_feat)

                #if 'train' in out_file:
                lib_out.write(str(labels[current_relation]) + ' ')
                #else:
                #    lib_out.write('0 ')
                lib_out.write(' '.join(i for i in sentence_feats if i) + '\n')
