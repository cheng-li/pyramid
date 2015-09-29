import logging, sys, os, numpy, ConfigParser, json
from word2vec import Word2Vec, Sent2Vec, LineSentence

""" Get the most useful N-Grams from the top features file """
def get_ngrams(file_path):
    # load the json content from the given file
    json_file = file(file_path)
    content = json.load(json_file)
    # parse the json data to get the grams
    ngrams = []
    for element in content:
        top_features = element.get('topFeatures')
        for feature in top_features:
            ngram = feature.get('ngram')
            ngrams.append(ngram)

    return ngrams


logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
logging.info("running %s" % " ".join(sys.argv))

# Read the configuration file
config = ConfigParser.ConfigParser()
config.read(sys.argv[1])

# Get the output directory and file paths
if config.has_option('Global', 'output.folder'):
    output_folder = config.get('Global', 'output.folder')
    train_sentences_path = os.path.join(output_folder, 'documents/train.sentences')
    test_sentences_path = os.path.join(output_folder, 'documents/test.sentences')
    if not os.path.isfile(train_sentences_path) or not os.path.isfile(test_sentences_path):
        logging.error("can't find training or testing sentences")
        sys.exit(0)

    word_vectors_model_path = os.path.join(output_folder, 'word.vector/word.vectors.model')
    if not os.path.isfile(word_vectors_model_path):
        logging.error("can't find the model file of word vector")
        sys.exit(0)

    # Create a directory and indicate file paths to store sentence vectors and the model
    if not os.path.exists(os.path.join(output_folder, 'sentence.vector')):
        os.makedirs(os.path.join(output_folder, 'sentence.vector'))
    train_vectors_path = os.path.join(output_folder, 'sentence.vector/train.vectors')
    test_vectors_path = os.path.join(output_folder, 'sentence.vector/test.vectors')
else:
    logging.error("please indicate the output folder in configuration file")
    sys.exit(0)


# Get the options from configuration file
if config.has_section('SentenceToVector'):
    alpha = config.getfloat('SentenceToVector', 'sent2vec.alpha')
    window = config.getint('SentenceToVector', 'sent2vec.window')
    sample = config.getfloat('SentenceToVector', 'sent2vec.sample')
    seed = config.getint('SentenceToVector', 'sent2vec.seed')
    workers = config.getint('SentenceToVector', 'sent2vec.workers')
    min_alpha = config.getfloat('SentenceToVector', 'sent2vec.min_alpha')
    sg = config.getint('SentenceToVector', 'sent2vec.sg')
    hs = config.getint('SentenceToVector', 'sent2vec.hs')
    negative = config.getint('SentenceToVector', 'sent2vec.negative')
    cbow_mean = config.getint('SentenceToVector', 'sent2vec.cbow_mean')
    iteration = config.getint('SentenceToVector', 'sent2vec.iteration')
    top_ngrams_on = config.getboolean('SentenceToVector', 'sent2vec.top_ngrams_on')

    # Get the top N-Grams from the top features file
    if top_ngrams_on:
        top_ngrams = get_ngrams(config.get('SentenceToVector', 'sent2vec.top_ngrams_file_path'))


    # Train sentence vectors on training and testing sets
    train_model = Sent2Vec(LineSentence(train_sentences_path), model_file=word_vectors_model_path, alpha=alpha,
                           window=window, sample=sample, seed=seed, workers=workers,min_alpha=min_alpha, sg=sg, hs=hs,
                           negative=negative, cbow_mean=cbow_mean, iteration=iteration, ngrams=top_ngrams)
    test_model = Sent2Vec(LineSentence(test_sentences_path), model_file=word_vectors_model_path, alpha=alpha,
                          window=window, sample=sample, seed=seed, workers=workers,min_alpha=min_alpha, sg=sg, hs=hs,
                          negative=negative, cbow_mean=cbow_mean, iteration=iteration, ngrams=top_ngrams)
else:
    # Train sentence vectors on training and testing sets
    train_model = Sent2Vec(LineSentence(train_sentences_path), model_file=word_vectors_model_path)
    test_model = Sent2Vec(LineSentence(test_sentences_path), model_file=word_vectors_model_path)

# Save the vectors into files
train_model.save_sent2vec_format(train_vectors_path)
test_model.save_sent2vec_format(test_vectors_path)

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)