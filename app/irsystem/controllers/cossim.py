import re
import numpy as np
import json
from nltk.tokenize import TreebankWordTokenizer


treebank_tokenizer = TreebankWordTokenizer()


def tokenize(text):
    """Returns a list of words that make up the text.

    Note: for simplicity, lowercase everything.
    Requirement: Use regular expressions to satisfy this function

    Params: {text: String}
    Returns: List
    """
    return list(filter(None, re.split('[^a-zA-Z]', text.lower())))


def build_inverted_index(msgs):
    """ Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    Note that doc_id refers to the index of the document/message in msgs.
    """
    inverted_dict = {}

    for i in range(len(msgs)):
        msg = msgs[i]['toks']
        msg_set = set(msg)
        for word in msg_set:
            count = msg.count(word)
            if word not in inverted_dict:
                inverted_dict[word] = [(i, count)]
            else:
                inverted_dict[word].append((i, count))

    return inverted_dict


def boolean_search(query_word, excluded_word, inverted_index):
    """ Search the collection of documents for the given query_word
        provided that the documents do not include the excluded_word

    Arguments
    =========

    query_word: string,
        The word we are searching for in our documents.

    excluded_word: string,
        The word excluded from our documents.

    inverted_index: an inverted index as above


    Returns
    =======

    results: list of ints
        Sorted List of results (in increasing order) such that every element is a `doc_id`
        that points to a document that satisfies the boolean
        expression of the query.

    """
    # initialize empty list
    merged_list_M = []

    # create sorted lists A (query word) and B (excluded word) of doc ids

    A_query = [x[0] for x in inverted_index[query_word.lower()]]
    B_excluded = [x[0] for x in inverted_index[excluded_word.lower()]]

    # start pointers at first elements
    A_i = 0
    B_j = 0

    # search algorithm
    while A_i < len(A_query) and B_j < len(B_excluded):
        if A_query[A_i] == B_excluded[B_j]:
            A_i += 1
            B_j += 1
        elif A_query[A_i] < B_excluded[B_j]:
            merged_list_M.append(A_query[A_i])
            A_i += 1
        else:
            B_j += 1

    # append rest of list A
    merged_list_M += A_query[A_i:]

    return merged_list_M


def compute_idf(inv_idx, n_docs, min_df=0, max_df_ratio=1):
    """ Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    Arguments
    =========

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = {}

    for word in inv_idx:
        dft = len(inv_idx[word])
        if dft >= min_df and dft/n_docs < max_df_ratio:
            idf[word] = np.log2(n_docs/(1+dft))

    return idf


def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    norms = np.zeros((n_docs,))

    for word in idf:
        sums = 0
        for loc in index[word]:
            norms[loc[0]] += (loc[1]*idf[word])**2

    return np.sqrt(norms)


def index_search(query, index, idf, doc_norms, tokenizer=treebank_tokenizer):
    """ Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score (descending order), but if there is
        a tie for the score, sort by the second element, that is
        the `doc_id` with ascending order.
        An example is as follows:

        score       doc_id
       [(0.9,       1000),
        (0.9,       1001),
        (0.8,       2000),
        (0.8,       2001),
        (0.8,       2002),
        ...]


    """
    query_toks = tokenizer.tokenize(query.lower())
    query_set = set(query_toks)
    query_u = list(query_set)

    # compute tf-idf vector q
    q = []
    for word in query_set:
        if word in idf:
            q.append(query_toks.count(word)*idf[word])
        else:
            q.append(0)

    # compute norm q
    q_norm = np.sqrt(np.sum(np.square(np.array(q))))

    # set up accumulators
    dict_results = {}
    for i in range(len(doc_norms)):
        dict_results[i] = 0

    # compute cossim
    for i in range(len(query_u)):
        if query_u[i] in idf:
            tups = index[query_u[i]]
            for tup in tups:
                idx = tup[0]
                dij = tup[1]*idf[query_u[i]]
                dict_results[idx] += q[i]*dij/(q_norm*doc_norms[idx])

    # convert into list of tups
    results = []
    for idx in dict_results:
        results.append((dict_results[idx], idx))

    # sort
    results = sorted(results, key=lambda x: x[1])
    results = sorted(results, key=lambda x: x[0], reverse=True)

    return results


#
