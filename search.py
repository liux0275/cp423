import get_index

# return the formatted top-5 docs by VSM ranking
def rank_by_VSM(query, doc_dict, tfidf_dict):
    # get the top 5 docs
    top_5 = get_index.vectorSpaceModel(query, doc_dict, tfidf_dict)
    
    outputs = "Using VSM to rank...\n"

    # format the outputs
    for filename, score in top_5.items():
        outputs += "document name -- {0}, score -- {1}\n".format(filename, score)
    return outputs

# return the formatted top-5 docs by BM25 ranking
def rank_by_BM25(query, doc_dict, tf_dict, clean_dict, df_dict, vocab):
    # get the top 5 docs
    bm25_dict = get_index.bm25(tf_dict, clean_dict, df_dict, vocab)
    top_5 = get_index.BM25Model(query, doc_dict, bm25_dict)
    

    outputs = "Using BM25 to rank...\n"

    # format the outputs
    for filename, score in top_5.items():
        outputs += "document name -- {0}, score -- {1}\n".format(filename, score)
    return outputs





