import logging, sys, os, ConfigParser
from word2vec import Word2Vec, LineSentence

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
    if not os.path.isfile(train_sentences_path):
        logging.error("can't find training sentences")
        sys.exit(0)

    # Create a directory and indicate file paths to store word vectors and the model
    if not os.path.exists(os.path.join(output_folder, 'word.vector')):
        os.makedirs(os.path.join(output_folder, 'word.vector'))
    word_vectors_path = os.path.join(output_folder, 'word.vector/word.vectors')
    word_vectors_model_path = os.path.join(output_folder, 'word.vector/word.vectors.model')
else:
    logging.error("please indicate the output folder in configuration file")
    sys.exit(0)

# Get the options from configuration file
if config.has_section('WordToVector'):
    size = config.getint('WordToVector', 'word2vec.size')
    alpha = config.getfloat('WordToVector', 'word2vec.alpha')
    window = config.getint('WordToVector', 'word2vec.window')
    min_count = config.getint('WordToVector', 'word2vec.min_count')
    sample = config.getfloat('WordToVector', 'word2vec.sample')
    seed = config.getint('WordToVector', 'word2vec.seed')
    workers = config.getint('WordToVector', 'word2vec.workers')
    min_alpha = config.getfloat('WordToVector', 'word2vec.min_alpha')
    sg = config.getint('WordToVector', 'word2vec.sg')
    hs = config.getint('WordToVector', 'word2vec.hs')
    negative = config.getint('WordToVector', 'word2vec.negative')
    cbow_mean = config.getint('WordToVector', 'word2vec.cbow_mean')

    # Train vectors for the words with the given options
    model = Word2Vec(LineSentence(train_sentences_path), size=size, alpha=alpha, window=window,
                     min_count=min_count, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                     sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean)
else:
    model = Word2Vec(LineSentence(train_sentences_path))

# Train word vectors using the given options
model.save(word_vectors_model_path)
model.save_word2vec_format(word_vectors_path)

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)