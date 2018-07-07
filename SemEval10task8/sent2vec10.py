import ast
import re
from parameters10 import *
from collections import Counter
import operator
if fire_tagger:
    from nltk.tag import pos_tag_sents
import string
import unicodedata
import pandas as pd


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


def read_tags(file):
    with open(file) as f:
        return {tag.strip(): i for i, tag in enumerate(f)}


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


def read_clusters(file):
    clusts = {}
    with open(file) as f:
        for line in f:
            wrd, clust = line.strip().split()
            if 'marlin' in file:
                clusts[wrd] = int(clust)
            elif 'brown' in file:
                clusts[wrd] = int(clust, 2)  # converts binary to int
    if 'brown' in file:
        clusts['<RARE>'] = max(clusts.values()) + 1
    return clusts


def read_embeddings(file):
    embs = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                continue
            else:
                word, vec = split[0], [float(val) for val in split[1:]]
                embs[word] = vec
    return embs


def read_cats(file):
    cats = {}
    with open(file) as f:
        for i, line in enumerate(f):
            cats[line.strip()] = i
    return cats


def create_pos_index(s_and_i, M, avg_m):
    posits = {}
    K = 2 * W + M
    vec_len = 2 * K + 2 + 1  # plus 2 for entities and plus 1 for middle
    for s, e1, e2 in s_and_i:
        norm_sent = normalize(M, [s, e1, e2], avg_m)
        vectors = vectorize(norm_sent, e1, e2, vec_len)
        for vector in vectors:
            if vector not in posits:
                posits[vector] = len(posits)
    posits['UNKNOWN'] = len(posits)
    return posits


def normalize(m, sent, avg_m):
    tkns, e1_index, e2_index = sent[0], sent[1], sent[2]
    l = e1_index - 1
    r = len(tkns) - e2_index
    current_m = e2_index - (e1_index + 1)
    if l == W and r == W:
        # print('1')
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l == W and r < W:
        # print('2')
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l == W and r > W:
        # print('3')
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r == W and l < W:
        # print('4')
        tkns = pad_left(tkns, l)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r == W and l > W:
        # print('5')
        tkns = slice_left(tkns, l)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l < W and r < W:
        # print('6')
        tkns = pad_left(tkns, l)
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l > W and r > W:
        # print('7')
        tkns = slice_left(tkns, l)
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l < W < r:
        # print('8')
        tkns = pad_left(tkns, l)
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r < W < l:
        # print('9')
        tkns = slice_left(tkns, l)
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    return tkns


def vectorize(tokes, e1, e2, vec_length):
    vecs = []
    middle = vec_length // 2
    for i in range(1, len(tokes) + 1):
        vec = [0 for _ in range(vec_length)]
        val1 = i - e1
        val2 = i - e2
        vec[middle + val1] = 1
        vec[middle + val2] = 1
        if vec not in vecs:
            vecs.append(vec)

    return [tuple(v) for v in vecs]


def pad_middle(lst, m, e1, e2):
    for i in range(m - (e2 - e1) + 1):
        if after_e1:
            lst.insert(W+1, None)
        elif before_e2:
            lst.insert(len(lst)-(W+1), None)
        elif not after_e1 and not before_e2:
            middle = m // 2
            if m % 2 == 0:
                lst.insert(W+middle+1, None)
            else:
                lst.insert(W+middle+2, None)  # +2 treats middle closer to e2 and +1 treats middle closer to e1
    return lst


def slice_middle(lst, avg_m, e1, e2):
    if after_e1:
        return lst[:e1+avg_m] + lst[e2:]
    if before_e2:
        return lst[:e1] + lst[e2-avg_m:]


def pad_left(lst, l):
    for i in range(W - l):
        lst.insert(0, None)
    return lst


def pad_right(lst, r):
    for i in range(W - r):
        lst.append(None)
    return lst


def slice_left(lst, l):
    return lst[l-W:]


def slice_right(lst, r):
    return lst[:-(r - W)]


if __name__ == '__main__':
    unknown = 'UNKNOWN'
    punc = set(string.punctuation)
    records_and_outs = [(path_to_feat_folder + 'record_train.txt', path_to_model_folder + 'libLinearInput_train.txt'),
                        (path_to_feat_folder + 'record_test.txt', path_to_model_folder + 'libLinearInput_test.txt')]
    num_words = 0
    num_positions = 0
    num_clusters = 0
    num_suffixes = 0
    num_shapes = 0
    num_tags = 0
    num_embeddings = 0
    char_emb_dims = 0
    cat_dims = 0

    if fire_words:
        words = read_feat_file(path_to_feat_folder + 'vocab.txt', unknown)
        num_words = len(words)
    if fire_clusters:
        if marlin:
            clusters = read_clusters(path_to_feat_folder + 'en_marlin_cluster_1000')
        elif brown:
            clusters = read_clusters(path_to_feat_folder + 'en_brown_1000')
        num_clusters = max(clusters.values()) + 1
    if fire_suffixes:
        suffixes = read_feat_file(path_to_feat_folder + 'suffixes.txt', unknown)
        num_suffixes = len(suffixes)
    if fire_shapes:
        shapes = read_shape_file(path_to_feat_folder + 'shapes.txt', unknown)
        num_shapes = len(shapes)
    if fire_tagger:
        tags = read_tags(path_to_feat_folder + 'tags.txt')
        num_tags = len(tags)
    if fire_embeddings:
        embeddings = read_embeddings(path_to_feat_folder + 'numberbatch-en.txt')
        num_embeddings = len(list(embeddings.values())[0])
    if fire_char_embeddings:
        char_embeddings = read_embeddings(path_to_feat_folder + 'numberbatch-en-char.txt')
        char_emb_dims = len(list(char_embeddings.values())[0])
    if fire_unicode_cats:
        unicode_categories = read_cats(path_to_feat_folder + 'categories.txt')
        cat_dims = len(unicode_categories)

    for record_file, out_file in records_and_outs:
        which = re.findall(r'record_(.*?).txt', record_file)[0]
        print('creating LibLinear ' + which + ' file...')
        records = read_record(record_file)
        sentences_and_indexes = [(record[1], int(record[2]), int(record[3])) for record in records]
        sentence_labels = [record[4] for record in records]
        with open(path_to_feat_folder + 'labels.txt') as labs:
            labels = {lab.strip(): i for i, lab in enumerate(labs)}

        if 'train' in record_file:
            all_M_vals = [record[6] for record in records]
            M = max(all_M_vals)
            avg_M = sum(all_M_vals) // len(records)
            if use_avg_M_plus_mode:
                M_counts = Counter(all_M_vals)
                avg_M += max(M_counts.items(), key=operator.itemgetter(1))[0]
            positions = create_pos_index(sentences_and_indexes, M, avg_M)
            num_positions = len(positions) if fire_positions else 0
        norm_sents = [normalize(M, sentence, avg_M) for sentence in sentences_and_indexes]
        if fire_tagger:
            tagged_sents = pos_tag_sents([[w if w is not None else 'none' for w in sent] for sent in norm_sents])
        else:
            tagged_sents = None
        if max_suffix_size == 0:
            word_lengths = [len(w) for w in words]
            suffix_length = sum(word_lengths) // num_words
        else:
            suffix_length = max_suffix_size
        num_char_embeddings = suffix_length * char_emb_dims
        num_cats = cat_dims * suffix_length
        len_token_vec = num_words + num_positions + num_clusters + num_suffixes + num_shapes + num_tags + num_embeddings + num_char_embeddings + num_cats
        feat_val = ':1.0'

        with open(out_file, 'w+') as lib_out:
            for i, sentence in enumerate(sentences_and_indexes):
                sentence_feats = []
                current_label = sentence_labels[i]
                norm_sent = norm_sents[i]
                if tagged_sents:
                    tagged_sent = tagged_sents[i]
                K = 2 * W + M
                pos_vecs = vectorize(norm_sent, sentence[1], sentence[2], 2 * K + 2 + 1)
                for idx, token in enumerate(norm_sent):
                    offset = idx * len_token_vec
                    token_feats = []
                    if token:
                        token_length = len(token)
                        if fire_words:
                            if token in words:
                                feat_pos = offset + words[token] + 1
                            else:
                                feat_pos = offset + words[unknown] + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_positions:
                            if pos_vecs[idx] in positions:
                                feat_pos = offset + positions[pos_vecs[idx]] + num_words + 1
                            else:
                                feat_pos = offset + positions[unknown] + num_words + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_clusters:
                            temp_token = ''.join('0' if char.isdigit() else char for char in token)
                            if any(char.isalpha() for char in temp_token) and len(temp_token) > 1:
                                temp_token = ''.join(char for char in temp_token if char not in string.punctuation)
                            if temp_token in clusters:
                                feat_pos = offset + clusters[temp_token] + num_words + num_positions + 1
                            else:
                                feat_pos = offset + clusters['<RARE>'] + num_words + num_positions + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_suffixes:
                            suffix_vec = []
                            for j in range(len(token)):
                                suffix = token[j:]
                                if suffix in suffixes:
                                    suffix_vec.append(suffixes[suffix])
                                else:
                                    if suffixes[unknown] not in suffix_vec:
                                        suffix_vec.append(suffixes[unknown])
                            for s in sorted(suffix_vec):
                                feat_pos = offset + s + num_words + num_positions + num_clusters + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                        if fire_shapes:
                            shape_vec = [0, 0, 0, 0, 0, 0, 0]
                            if any(char.isupper() for char in token):
                                shape_vec[0] = 1
                            if '-' in token:
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
                            feat_pos = offset + shapes[tup_vec] + num_words + num_positions + num_clusters + num_suffixes + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_tagger:
                            current_tag = tagged_sent[idx][1]
                            feat_pos = offset + tags[current_tag] + num_shapes + num_words + num_positions + num_clusters + num_suffixes + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_embeddings:
                            temp_token = ''.join('#' if char.isdigit() else char for char in token)
                            temp_token = ''.join('_' if char == '-' else char for char in temp_token)
                            lowered = temp_token.lower()
                            if lowered in embeddings:
                                vec = embeddings[lowered]
                                token_feats += [str(offset + n + num_tags + num_shapes + num_words + num_positions +
                                                num_clusters + num_suffixes + 1) + ':' + str(vec[n])
                                                for n in range(num_embeddings)]
                        if fire_char_embeddings:
                            vec = []
                            empty_vec = [0.0] * char_emb_dims
                            if token_length < suffix_length:
                                for i in range(suffix_length - token_length):
                                    vec += empty_vec
                            suffix = token[-suffix_length:]
                            for char in suffix:
                                if char in char_embeddings:
                                    vec += char_embeddings[char]
                                else:
                                    print(char, 'not found')
                                    vec += empty_vec
                            token_feats += [str(offset + n + num_embeddings + num_tags + num_shapes + num_words
                                                + num_positions + num_clusters + num_suffixes + 1) + ':' + str(vec[n])
                                                for n in range(char_emb_dims)]
                        if fire_unicode_cats:
                            cat_vec = []
                            suffix = token[-suffix_length:]
                            for i, char in enumerate(suffix):
                                cat = unicodedata.category(char)
                                cat_vec.append(unicode_categories[cat] + i*cat_dims)
                            for c in sorted(cat_vec):
                                feat_pos = offset + c + num_char_embeddings + num_embeddings + num_tags + num_shapes + num_words + num_positions + num_clusters + num_suffixes + 1
                                char_feat = str(feat_pos) + feat_val
                                token_feats.append(char_feat)

                    else:
                        unknown_word = None
                        unknown_position = None
                        unknown_cluster = None
                        unknown_suffix = None
                        unknown_shape = None
                        unknown_tag = None
                        # not handling unknown embeddings because they don't have indices
                        if fire_words:
                            feat_pos = offset + words[unknown] + 1
                            unknown_word = str(feat_pos) + feat_val
                        if fire_positions:
                            feat_pos = offset + positions[unknown] + num_words + 1
                            unknown_position = str(feat_pos) + feat_val
                        if fire_clusters:
                            feat_pos = offset + clusters['<RARE>'] + num_words + num_positions + 1
                            unknown_cluster = str(feat_pos) + feat_val
                        if fire_suffixes:
                            feat_pos = offset + suffixes[unknown] + num_words + num_positions + num_clusters + 1
                            unknown_suffix = str(feat_pos) + feat_val
                        if fire_shapes:
                            feat_pos = offset + shapes[unknown] + num_words + num_positions + num_clusters + num_suffixes + 1
                            unknown_shape = str(feat_pos) + feat_val
                        if fire_tagger:
                            feat_pos = offset + tags['NN'] + num_shapes + num_words + num_positions + num_clusters + num_suffixes + 1
                            unknown_tag = str(feat_pos) + feat_val
                        token_feats = [unknown_word, unknown_position, unknown_cluster, unknown_suffix, unknown_shape,
                                       unknown_tag]

                    sentence_feats += token_feats

                lib_out.write(str(labels[current_label]) + ' ')
                lib_out.write(' '.join(i for i in sentence_feats if i) + '\n')