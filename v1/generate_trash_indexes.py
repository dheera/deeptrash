#!/usr/bin/env python3
import json
from nltk.corpus import wordnet

def get_hyponyms(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())

with open('trash_mapping.json', 'r') as f:
    mapping = json.loads(f.read())

with open('Inception-labels.txt', 'r') as f:
    labels = [int(l.split(' ')[0].strip('n')) for l in f]

trash_indexes = {}

for trash_category in mapping:
    print(trash_category)
    synsets = set()
    for synset_name in mapping[trash_category]:
        s = wordnet.synset(synset_name)
        synsets.add(s.offset())
        for hyponym in get_hyponyms(s):
             synsets.add(hyponym.offset())
    indexes = []
    for synset in synsets:
        if synset in labels:
            indexes.append(labels.index(synset))

    trash_indexes[trash_category] = indexes


with open('trash_indexes.json', 'w') as f:
    f.write(json.dumps(trash_indexes))
