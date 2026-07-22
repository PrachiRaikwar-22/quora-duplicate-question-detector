import numpy as np
import fuzzywuzzy.fuzz as fuzz
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')

FILE_ID = '1LyedO-67tQDTjSLYuYv8aXadnOcWKxWw'
MODEL_PATH = 'rf_model.pkl'

STOP_WORDS = set(stopwords.words("english"))

def get_average_w2v(sentence, model, vector_size=100):
    words = sentence.lower().split()
    valid_words = [word for word in words if word in model.wv]
    if len(valid_words) == 0:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_words], axis=0)

def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1) + len(w2)

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    length_features[2] = abs(len(q1) - len(q2))
    return length_features

def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

def extract_all_features(q1, q2, model):
    input_query = []
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    cw = test_common_words(q1, q2)
    tw = test_total_words(q1, q2)
    input_query.append(cw)
    input_query.append(tw)
    input_query.append(round(cw / (tw + 0.0001), 2))

    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))
    features_22 = np.array(input_query).reshape(1, 22)

    v1 = get_average_w2v(q1, model).reshape(1, -1)
    v2 = get_average_w2v(q2, model).reshape(1, -1)

    sim = cosine_similarity(v1, v2)[0][0] if (np.any(v1) and np.any(v2)) else 0.0
    sim_feature = np.array([[sim]])

    return np.hstack((features_22, v1, v2, sim_feature))

def query_point_creator(q1, q2, w2v_model):
    return extract_all_features(q1, q2, w2v_model)
