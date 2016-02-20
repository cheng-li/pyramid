package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.node.Node;
import org.elasticsearch.search.SearchHit;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by Jilin on 9/12/15.
 */
class Document {
    public String id;
    public String body;
    public String label;
    public String split;

    public Document(String id, String body, String label, String split) {
        this.id = id;
        this.body = body;
        this.label = label;
        this.split = split;
    }

    public String toString() {
        return "Document:\n"
                + "\tid: " + this.id + "\n"
                + "\tbody: " + this.body + "\n"
                + "\tlabel: " + this.label + "\n"
                + "\tsplit: " + this.split;
    }
}


public class Exp300 {

    static Config config = null;

    public static void main(String[] args) throws Exception {
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("es.on")) {
            // Get sentences and labels from ElasticSearch.
            System.out.println("Reading documents from Elastic Search...");
            getSentencesAndLabelsFromElasticSearch();
            System.out.println("The sentences and labels have been written into files!");
        }

        if (config.getBoolean("word2vec.on")) {
            // Train word vectors using the sentences and labels
            System.out.println("Start to train word vectors...");
            wordsToVectors(Paths.get(args[0]).toAbsolutePath().normalize().toString());
            System.out.println("Finish to train word vectors!");
        }

        if (config.getBoolean("sent2vec.on")) {
            // Train sentence vectors for training and testing set
            System.out.println("Start to train sentence vectors...");
            sentencesToVectors(Paths.get(args[0]).toAbsolutePath().normalize().toString());
            System.out.println("Finish to train sentence vectors!");
        }

        // Populate the vectors and labels into trec files only when sent2vec.on = true
        // to avoid FileNotFoundException.
        if (config.getBoolean("sent2vec.on")) {
            System.out.println("Writing vectors and labels into Trec files...");
            writeTrec("/sentence.vector/train.vectors", "/documents/train.labels", "/trec/train.trec");
            writeTrec("/sentence.vector/test.vectors", "/documents/test.labels", "/trec/test.trec");
            System.out.println("Finished!");
    
            similarity(Paths.get(args[0]).toAbsolutePath().normalize().toString(), "good", "great");
            mostSimilar(Paths.get(args[0]).toAbsolutePath().normalize().toString(), "great", 10);
        }
    }

    /**
     * Get all of the documents in Elastic Search with the given configuration
     * @return a list of documents
     */
    static ArrayList<Document> getAllDocuments() {
        String clusterName = config.getString("es.clusterName");
        String indexName = config.getString("es.indexName");
        String documentType = config.getString("es.documentType");

        // Split the given labels and populate them into a HashMap
        int numberOfClasses = config.getInt("es.numberOfClasses");
        HashMap<String, String> labelMap = new HashMap<>();
        for (int i = 0; i < numberOfClasses; i++) {
            labelMap.put(config.getString("es.label" + Integer.toString(i)), Integer.toString(i));
        }

        Node node = nodeBuilder().client(true).clusterName(clusterName).node();
        Client client = node.client();

        // Get all the documents and add them into the list.
        ArrayList<Document> documents = new ArrayList<>();
        int scrollSize = 1000, counter = 0;
        SearchResponse response = null;
        while (response == null || response.getHits().hits().length != 0) {
            response = client.prepareSearch(indexName)
                    .setTypes(documentType)
                    .setFrom(counter * scrollSize)
                    .setSize(scrollSize)
                    .execute()
                    .actionGet();

            for (SearchHit hit : response.getHits()) {
                String id = hit.getId();
                Map<String, Object> source = hit.getSource();
                String body = (String)source.get("body");
                String label = source.get("label").toString();
                String split = (String)source.get("split");

                if (labelMap.containsKey(label)) {
                    documents.add(new Document(id, body, labelMap.get(label), split));
                }
            }
            counter++;
        }

        node.close();
        return documents;
    }

    static BufferedWriter getBufferedWriter(String outputFolder, String filePath) throws IOException {
        File file = new File(outputFolder, filePath);
        file.getParentFile().mkdirs();
        return new BufferedWriter(new FileWriter(file));
    }

    static BufferedReader getBufferedReader(String outputFolder, String filePath) throws IOException {
        File file = new File(outputFolder, filePath);
        return new BufferedReader(new FileReader(file));
    }

    /**
     * Get all of the documents in Elastic Search, and split them into two
     * sets: train and test. Then write the sentences and labels into files.
     */
    static void getSentencesAndLabelsFromElasticSearch() throws IOException {
        ArrayList<Document> documents = getAllDocuments();

        String outputFolder = config.getString("output.folder");
        BufferedWriter trainSentencesWriter = getBufferedWriter(outputFolder, "/documents/train.sentences");
        BufferedWriter trainLabelsWriter = getBufferedWriter(outputFolder, "/documents/train.labels");
        BufferedWriter testSentencesWriter = getBufferedWriter(outputFolder, "/documents/test.sentences");
        BufferedWriter testLabelsWriter = getBufferedWriter(outputFolder, "/documents/test.labels");

        // Shuffle the list and split the documents into two sets according to the given percentage.
        // Otherwise, just split them into two sets according to the split field.
        if (config.getBoolean("es.shuffle")) {
            Collections.shuffle(documents);
            int totalNumber = documents.size();
            int numberOfTrainingPoints = (int)(totalNumber * config.getDouble("es.trainPercentage"));
            for (int i = 0; i < numberOfTrainingPoints; i++) {
                trainSentencesWriter.write(documents.get(i).body + "\n");
                trainLabelsWriter.write(documents.get(i).label + "\n");
            }
            for (int i = numberOfTrainingPoints; i < totalNumber; i++) {
                testSentencesWriter.write(documents.get(i).body + "\n");
                testLabelsWriter.write(documents.get(i).label + "\n");
            }
        } else {
            for (Document doc : documents) {
                if (doc.split.equals("train")) {
                    trainSentencesWriter.write(doc.body + "\n");
                    trainLabelsWriter.write(doc.label + "\n");
                } else {
                    testSentencesWriter.write(doc.body + "\n");
                    testLabelsWriter.write(doc.label + "\n");
                }
            }
        }

        trainSentencesWriter.close();
        trainLabelsWriter.close();
        testSentencesWriter.close();
        testLabelsWriter.close();
    }

    /**
     * Execute the given command line and print out the outputs or errors
     * @param command given command line
     * @throws IOException
     */
    static void exec(String command) throws IOException {
        Process proc = Runtime.getRuntime().exec(command);

        String line = null;
        // read the output from the command
        BufferedReader stdInput = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        while ((line = stdInput.readLine()) != null) {
            System.out.println(line);
        }

        // read any errors from the attempted command1
        BufferedReader stdError = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
        while ((line = stdError.readLine()) != null) {
            System.out.println(line);
        }
    }

    /**
     * Execute python methods to train word vectors using the given options
     */
    static void wordsToVectors(String configFilePath) throws IOException {
        String pythonFolder = config.getString("python.folder");
        String command = String.format("python %s/word_to_vector.py %s", pythonFolder, configFilePath);
        exec(command);
    }

    /**
     * Execute python methods to train sentence vectors using the given options
     */
    static void sentencesToVectors(String configFilePath) throws IOException {
        String pythonFolder = config.getString("python.folder");
        String command = String.format("python %s/sentence_to_vector.py %s", pythonFolder, configFilePath);
        exec(command);
    }

    /**
     *  Populate the given vectors and labels into trec files
     */
    static void writeTrec(String vectorsFilePath,
                          String labelsFilePath, String trecFilePath) throws Exception {
        String outputFolder = config.getString("output.folder");
        int numberOfFeatures = config.getInt("word2vec.size");
        int numberOfClasses = config.getInt("es.numberOfClasses");

        BufferedReader trainReader = getBufferedReader(outputFolder, vectorsFilePath);
        BufferedReader labelReader = getBufferedReader(outputFolder, labelsFilePath);
        ArrayList<Integer> labels = new ArrayList<>();
        String labelLine = null;
        while ((labelLine = labelReader.readLine()) != null) {
            labels.add(Integer.valueOf(labelLine));
        }

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(labels.size())
                .numFeatures(numberOfFeatures)
                .numClasses(numberOfClasses)
                .dense(true)
                .missingValue(false)
                .build();

        for (int i = 0; i < labels.size(); i++) {
            String line = trainReader.readLine();
            String[] features = line.split(" ");
            if (features.length != numberOfFeatures) {
                throw new Exception("The number of features is invalid");
            }

            for (int j = 0; j < numberOfFeatures; j++) {
                dataSet.setFeatureValue(i, j, Double.valueOf(features[j]));
            }

            int label = labels.get(i);
            dataSet.setLabel(i, label);
        }

        TRECFormat.save(dataSet, new File(outputFolder, trecFilePath));
    }

    /**
     * Compute cosine similarity between two words
     */
    static void similarity(String configFilePath, String w1, String w2) throws IOException {
        String pythonFolder = config.getString("python.folder");
        String command = String.format("python %s/similarity.py %s %s %s", pythonFolder, configFilePath, w1, w2);
        exec(command);
    }

    /**
     * Find the top-N most similar words
     */
    static void mostSimilar(String configFilePath, String w, int topn) throws IOException {
        String pythonFolder = config.getString("python.folder");
        String command = String.format("python %s/most_similar.py %s %s %d", pythonFolder, configFilePath, w, topn);
        exec(command);
    }
}
