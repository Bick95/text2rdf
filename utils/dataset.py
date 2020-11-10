#!/usr/bin/env python
# coding: utf-8

# Imports
# General purpose
import os
import glob
import random
import xml.etree.ElementTree as ET

# Set paths where to retrieve data from
DS_BASE_PATH = './WebNLG/'

TRAIN_PATH = DS_BASE_PATH + 'train/'
TEST_PATH = DS_BASE_PATH + 'dev/'

originaltripleset_index = 0     # Index of 'originaltripleset' attribute in XML entry
modifiedtripleset_index = 1     # Index of 'modifiedtripleset' attribute in XML entry
first_lexical_index = 2         # Index as of which verbalizations of RDF triples start in entry


# Train Data
def get_train_vocab(min_num_triples, max_num_triples):
    train_dirs = [TRAIN_PATH + str(i) + 'triples/' for i in range(min_num_triples, max_num_triples + 1)]
    print('Train dirs:', train_dirs)

    # Usage of train: train[target_nr_triples][element_id]['target_attribute']
    train = [[] for i in range(min_num_triples, max_num_triples + 1)]

    # Documents how many entries there are per number of triples
    train_stats = [0 for i in range(min_num_triples, max_num_triples + 1)]

    # Iterate through all files per number of triples and per category and load data
    for i, d in enumerate(train_dirs):
        nr_triples = list(range(min_num_triples, max_num_triples + 1))[i]

        for filename in glob.iglob(d + '/**', recursive=False):
            if os.path.isfile(filename):  # Filter dirs

                tree = ET.parse(filename)
                root = tree.getroot()

                entries = root[0]
                train_stats[nr_triples - min_num_triples] += len(entries)

                for entry in entries:

                    modified_triple_set = entry[modifiedtripleset_index]
                    unified_triple_set = []

                    for triple in modified_triple_set:
                        # Make a list containing a conjunction of all individual triples
                        triple_list = [x.strip() for x in triple.text.split('|')]
                        unified_triple_set += triple_list

                    verbalizations = entry[first_lexical_index:]

                    for verbalization in verbalizations:
                        if verbalization.text.strip() == '':
                            continue

                        train[i].append({'category': entry.attrib['category'],
                                         'id': entry.attrib['eid'],
                                         'triple_cnt': nr_triples,
                                         'text': verbalization.text,
                                         'triple': unified_triple_set,
                                         })

    print(train)
    print(train_stats)
    return train, train_stats


# Test Data
def get_test_vocab(min_num_triples, max_num_triples):
    train_dirs = [TEST_PATH + str(i) + 'triples/' for i in range(min_num_triples, max_num_triples + 1)]
    print('Test  dirs:', train_dirs)

    # Usage of test: test[target_nr_triples][element_id]['target_attribute']
    test = [[] for i in range(min_num_triples, max_num_triples + 1)]

    # Documents how many entries there are per number of triples
    test_stats = [0 for i in range(min_num_triples, max_num_triples + 1)]

    # Iterate through all files per number of triples and per category and load data
    for i, d in enumerate(train_dirs):
        nr_triples = list(range(min_num_triples, max_num_triples + 1))[i]

        for filename in glob.iglob(d + '/**', recursive=False):
            if os.path.isfile(filename):  # Filter dirs

                tree = ET.parse(filename)
                root = tree.getroot()

                entries = root[0]
                test_stats[nr_triples - min_num_triples] += len(entries)

                for entry in entries:

                    modified_triple_set = entry[modifiedtripleset_index]
                    unified_triple_set = []

                    for triple in modified_triple_set:
                        # Make a list containing a conjunction of all individual triples
                        triple_list = [x.strip() for x in triple.text.split('|')]
                        unified_triple_set += triple_list

                    verbalizations = entry[first_lexical_index:]

                    for verbalization in verbalizations:
                        if verbalization.text.strip() == '':
                            continue

                        test[i].append({'category': entry.attrib['category'],
                                        'id': entry.attrib['eid'],
                                        'triple_cnt': nr_triples,
                                        'text': verbalization.text,
                                        'triple': unified_triple_set,
                                        })

    print(test)
    print(test_stats)
    return test, test_stats


# Spilt Train Data into Train and Dev (for intermindiate validation throughout training)
def get_dev_vocab(train, train_stats, min_num_triples, max_num_triples, dp=0.15):
    # Percentage of train data reserved for validation throughout training
    dev_percentage = dp

    # Init dev dataset
    dev = [[] for i in range(min_num_triples, max_num_triples + 1)]

    # Sample number of dev instances per number of triples
    dev_stats = [int(dev_percentage * train_stats[i]) for i in range(0, max_num_triples + 1 - min_num_triples)]

    print('Samples per nr of triples:', dev_stats)

    # Sample indices to be reserved for dev dataset for each nr of triples
    dev_indices = [random.sample(range(0, len(train[i])), dev_stats[i]) for i in
                   range(0, max_num_triples + 1 - min_num_triples)]
    for i in range(len(dev_indices)):
        dev_indices[i].sort(reverse=True)

    # Copy selected dev-entries into dev & delete all duplicates/related entries from train dataset
    for nr_triples in range(0, max_num_triples + 1 - min_num_triples):

        # Iterate through all indices reserved for validation set (per nr of triples)
        for index in dev_indices[nr_triples]:

            # Select index'th train entry (to become dev/validation data)
            selected_entry = train[nr_triples][index]

            # Extract indentifying attributes
            entrys_category = selected_entry['category']
            entrys_idx = selected_entry['id']

            # Put selected entry into dev set
            dev[nr_triples].append(selected_entry)

            # Find all entries of matching index & category and remove them from train data
            for entry in train[nr_triples]:
                if entry['id'] == entrys_idx and entry['category'] == entrys_category:
                    train[nr_triples].remove(entry)

    print(dev)
    print(dev_stats)
    return train, train_stats, dev, dev_stats


# Print Stats
def print_stats(train, dev, test, min_num_triples, max_num_triples):
    print('Minimal number of triples:', min_num_triples)
    print('Maximal number of triples:', max_num_triples)

    print()

    print('Training: ')
    for nr_triples in range(min_num_triples, max_num_triples + 1):
        print('Given %i triples per sentence: ' % nr_triples)
        print('Number of combinations of triples and verbalizations:', len(train[nr_triples - min_num_triples]))

    print()

    print('Dev: ')
    for nr_triples in range(min_num_triples, max_num_triples + 1):
        print('Given %i triples per sentence: ' % nr_triples)
        print('Number of combinations of triples and verbalizations:', len(dev[nr_triples - min_num_triples]))

    print()

    print('Testing: ')
    for nr_triples in range(min_num_triples, max_num_triples + 1):
        print('Given %i triples per sentence: ' % nr_triples)
        print('Number of combinations of triples and verbalizations:', len(test[nr_triples - min_num_triples]))

