from pandas import read_csv
import numpy as np
import nltk
from re import sub
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from os.path import exists
import pickle
from keras import backend as K


class Constants:
	lstm_model_name = "lstm_model"
	gensim_model_name = 'gensim_model'


def get_list_words(essay):
    """
    This removes all the stopwords in the essay and also remove words which are not alphabetical.
    """
    # replacing all the words which are not in [a-zA-Z] with a blank space.
    essay = sub('[^A-Za-z]', " ", essay)

    # converting essay to lower case in order to have a uniform and similar results accross all the essays.
    lower_essay = essay.lower()

    words = lower_essay.split()

    # removing all the stopwords from essay vector so that, only specific word vectors remain.
    stops = set(stopwords.words("english"))
    words = [word for word in words if word not in stops]

    # note: the output is a tuple.
    return (words)


def essay_to_sentences(essay: str):
    """
    Converts the essays to sentences followed by word tokenization by above defined method.
    @essay: a single big of essay
    """

    # removing any blank space and newline if any from essay.
    essay = essay.strip()

    # Getting default nltk tokenizer from nltk library.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # converting essay string to sentences.
    sentence_strings = tokenizer.tokenize(essay)

    # for each sentence in tokenized sentences, get corresponding word vector and append it to sentence list.
    sentences = []
    for sentence_string in sentence_strings:
        if len(sentence_string) > 0:
            sentences.append(get_list_words(sentence_string))
    return sentences


def get_number_vector_from_wordVector(words, model, vector_size):
    """Make Feature Vector from the words list of an Essay."""

    # initialising an empty array of 0's.
    word_vector = np.zeros((vector_size,), dtype="float32")

    # number of words that are there in the set of final vector.
    num_words = 0

    # this is mapping of index of words that are vectorized in the model while trainining.
    index_to_word_set = set(model.wv.index2word)

    # check if the word is a common word and it exists in the model dataset so that it can be vectorised.
    for word in words:
        if word in index_to_word_set:
            num_words += 1  # keeping a count of number of elements in the vector so that,
            # It can be normalised.

            word_vector = np.add(word_vector, model[word])  # This is better version of extending a list.

    # Normalise the 
    word_vector = np.divide(word_vector, num_words)

    # not consuming the data as it is in order to perform some analysis on this vector.
    return word_vector


def get_avg_feature_vectors(essays, model, vector_size):
    """
    Combining all the functions defined to make a single function 
    that will return list of number vector from the essay string for all the rows in the dataset.
    """

    # creating an empty zero vector for fillin it with appropriate numbers.
    # using float32 reduces the space required for each vector. 
    essay_feature_vecs = np.zeros((len(essays), vector_size), dtype="float32")

    # iterate over all the the essays.
    for counter, essay in enumerate(essays):
        # generating number vector from the essay.
        essay_feature_vecs[counter] = get_number_vector_from_wordVector(essay, model, vector_size)

    return essay_feature_vecs


def get_model(vector_size=500, print_summary=True):
    model = Sequential()

    # Input layer.
    model.add(LSTM(vector_size, input_shape=[1, vector_size], return_sequences=True))

    # lstm layer.
    model.add(LSTM(256))

    # added to reduce any over-fitting of data.
    model.add(Dropout(0.5))

    # First dense layer.
    model.add(Dense(1, activation='relu'))

    # adding hyper-parameters to model.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    if print_summary:
        print(model.summary())

    return model


def predict_single_essay(essay, gensim_model_name, lstm_model_name, gensim_model=None, lstm_model=None):
    if gensim_model is None:
        gensim_model = load_gensim_model(gensim_model_name)
    if lstm_model is None:
        lstm_model = load_model(lstm_model_name)
    cleant_essay = [get_list_words(essay)]
    features = np.array(get_avg_feature_vectors(cleant_essay, model=gensim_model, vector_size=300))
    prediction = np.around(np.array(lstm_model.predict(features.reshape(features.shape[0], 1, features.shape[1]))[-1]).reshape(1)[0])
    K.clear_session()
    return prediction


def load_gensim_model(file_name):
    with open(file_name, 'rb') as model_file:
        return pickle.load(model_file)


def save_gensim_model(model, model_name):
    with open(model_name, 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    if True:
        lstm_model_name = Constants.lstm_model_name
        gensim_model_name = Constants.gensim_model_name
        #../dataset/dataset/
        data = read_csv('training.tsv', encoding='latin', sep='\t')
        train_data = data[:8 * len(data['essay']) // 10]
        test_data = data[8 * len(data['essay']) // 10:]
        del data  # Freeing up memory.
        train_essays = train_data['essay']
        test_essays = test_data['essay']
        y_train = train_data['rater1_domain1']
        y_test = test_data['rater1_domain1']

        vector_size = 300
        min_word_count = 40
        num_workers = 4
        context = 10
        down_sampling = 1e-3

    if exists(lstm_model_name) and exists(gensim_model_name):
        print("Loading found models from local machine.")
        lstm_model = load_model(lstm_model_name)
        gensim_model = load_gensim_model(gensim_model_name)
        print(predict_single_essay(
            "Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.",
            gensim_model_name=gensim_model_name, gensim_model=gensim_model, lstm_model=lstm_model,
            lstm_model_name=lstm_model_name))
    else:
        sentences = []
        print("No models were found, training new model.")
        for essay in train_essays:
            sentences += essay_to_sentences(essay)
        model = Word2Vec(sentences, workers=num_workers, size=vector_size, min_count=min_word_count, window=context,
                         sample=down_sampling)

        # init_sims is generally used to normalise all the doc vectors in the model to same length and range of value.
        model.init_sims(replace=True)
        save_gensim_model(model, gensim_model_name)

        clean_train_essays = [get_list_words(essay_v) for essay_v in train_essays]
        clean_test_essays = [get_list_words(essay_v) for essay_v in test_essays]

        train_data_vectors = get_avg_feature_vectors(clean_train_essays, model, vector_size)
        test_data_vectors = get_avg_feature_vectors(clean_test_essays, model, vector_size)

        train_data_vectors = np.array(train_data_vectors)
        test_data_vectors = np.array(test_data_vectors)

        # Reshaping train and test vectors to 3 dimensions.
        train_data_vectors = train_data_vectors.reshape((train_data_vectors.shape[0], 1, train_data_vectors.shape[1]))
        test_data_vectors = np.reshape(test_data_vectors, (test_data_vectors.shape[0], 1, test_data_vectors.shape[1]))

        lstm_model = get_model(vector_size)
        lstm_model.fit(train_data_vectors, y_train, batch_size=150, epochs=60)

        lstm_model.save(lstm_model_name)
        y_pred = np.around(lstm_model.predict(test_data_vectors))

        print("R2 score:", r2_score(y_test.values, lstm_model.predict(test_data_vectors)))
        print('mean squared error:', mean_squared_error(y_pred, y_test))
