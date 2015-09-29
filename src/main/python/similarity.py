import sys, os, ConfigParser
from word2vec import Word2Vec

# Read the configuration file
config = ConfigParser.ConfigParser()
config.read(sys.argv[1])

# Get the output directory and file paths
if config.has_option('Global', 'output.folder'):
    output_folder = config.get('Global', 'output.folder')

    word_vectors_model_path = os.path.join(output_folder, 'word.vector/word.vectors.model')
    if not os.path.isfile(word_vectors_model_path):
        print "can't find the model file of word vector"
        sys.exit(0)

model = Word2Vec.load(word_vectors_model_path)
print "The similarity between %s and %s is %f" % (sys.argv[2], sys.argv[3], model.similarity(sys.argv[2], sys.argv[3]))