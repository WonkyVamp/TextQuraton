
import os
import pickle

import gensim.models.doc2vec
import joblib

# invoice_gokul_final
from invoice_gokul_final.nlp.en.segments.sentences import get_sentence_list
from invoice_gokul_final.nlp.en.tokens import get_stem_list


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

d2v_model_filename = "d2v_all_size100_window10.model"
d2v_model_path = os.path.join(data_dir, d2v_model_filename)

if os.path.exists(d2v_model_path):
    d2v_model = gensim.models.doc2vec.Doc2Vec.load(d2v_model_path)
else:
    d2v_model_filenames = sorted([i for i in os.listdir(data_dir)
                                  if i.startswith('{}.part.'.format(d2v_model_filename))])
    if not d2v_model_filenames:
        raise RuntimeError('Doc2Vec model file "{}" not found'.format(d2v_model_filename))
    d2v_model_pickled = b''
    for filename in d2v_model_filenames:
        d2v_model_pickled += open(os.path.join(data_dir, filename), 'rb').read()
    d2v_model = pickle.loads(d2v_model_pickled)


rf_model = joblib.load(
    os.path.join(data_dir, "is_contract_classifier.pickle"))


def process_sentence(sentence):
    return [s for s in get_stem_list(sentence, stopword=True, lowercase=True) if s.isalpha()]


def process_document(document):
    doc_words = []
    for sentence in get_sentence_list(document):
        doc_words.extend(process_sentence(sentence))
    return doc_words


def is_contract(text, min_probability=0.5, return_probability=False):
    text_vector = d2v_model.infer_vector(process_document(text))

    try:
        classifier_score = rf_model.predict_proba([text_vector])[0, 1]
    except IndexError:
        return None

    ret = bool(classifier_score >= min_probability)

    if return_probability:
        ret = (ret, classifier_score)

    return ret
