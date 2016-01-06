package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;

import java.io.*;
import java.util.*;

/**
 * Created by Rainicy on 1/5/16.
 */
public class Exp219 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        String output = config.getString("output");
        String modelName = config.getString("modelName");

        String path = output + "/" + modelName;

        int startIter = config.getInt("startIter");
        int endIter = config.getInt("endIter");

        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.Data"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        MultiLabel[] labels = dataSet.getMultiLabels();


        for (int i=startIter; i<endIter+1; i++) {
            System.out.println("processing: " + path + " - iter: " + i);
            BMMClassifier bmmClassifier = loadModel(config, i);
            MultiLabel[] predictions = bmmClassifier.predict(dataSet);
            double[][] gammas = getGammas(config, i, "gammas");
            double[][] PIs = getGammas(config, i, "PIs");

            Map<Double, HashMap<Boolean, Integer>> gammasPerplexityMap = getPerplexity(labels, predictions, gammas);
            Map<Double, HashMap<Boolean, Integer>> PIsPerplexityMap = getPerplexity(labels, predictions, PIs);

            writePerplexityFile(gammasPerplexityMap, path, i, "gammas");
            writePerplexityFile(gammasPerplexityMap, path, i, "PIs");
        }

    }

    private static void writePerplexityFile(Map<Double, HashMap<Boolean,Integer>> perplexityMap, String path,
                                            int iter, String type) throws IOException {
        File file = new File(path, "iter."+iter+"."+type+".perplexity");
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        for (Map.Entry<Double, HashMap<Boolean, Integer>> entry : perplexityMap.entrySet()) {
            double perplexity = entry.getKey();
            HashMap<Boolean, Integer> hMap = entry.getValue();
            int falseCount = hMap.containsKey(false) ? hMap.get(false) : 0;
            int trueCount = hMap.containsKey(true) ? hMap.get(true) : 0;

            bw.write(perplexity + "\t" + falseCount + "\t" + trueCount + "\n");
        }
        bw.close();
    }

    private static Map<Double, HashMap<Boolean, Integer>> getPerplexity(MultiLabel[] labels, MultiLabel[] predictions, double[][] gammas) {
        Map<Double, HashMap<Boolean, Integer>> perplexityMap = new TreeMap<>();

        for (int i=0; i<gammas.length; i++) {
            double entropy = 0.0;
            for (int j=0; j<gammas[i].length; j++) {
                double p = gammas[i][j];
                entropy -= p * Math.log(p)/Math.log(2);
            }
            double perplexity = Math.pow(2, entropy);
            boolean isMatch = labels[i].equals(predictions[i]);
            if (!perplexityMap.containsKey(perplexity)) {
                perplexityMap.put(perplexity, new HashMap<>());
            }
            if (!perplexityMap.get(perplexity).containsKey(isMatch)) {
                perplexityMap.get(perplexity).put(isMatch, 1);
            } else {
                perplexityMap.get(perplexity).put(isMatch, perplexityMap.get(perplexity).get(isMatch)+1);
            }
        }

        return perplexityMap;
    }

    private static double[][] getGammas(Config config, int iter, String type) throws IOException {
        List<ArrayList<Double>> listGammas = new ArrayList<>();
        String path = config.getString("output") + "/" + config.getString("modelName");
        File gammaFile = new File(path, "iter." + iter + "." + type);

        BufferedReader br = new BufferedReader(new FileReader(gammaFile));
        String line;
        while ((line = br.readLine()) != null) {
            String[] stringGammas = line.split("\\t");
            ArrayList<Double> list = new ArrayList<>();
            for (int i=0; i<stringGammas.length; i++) {
                list.add(Double.parseDouble(stringGammas[i]));
            }
//            System.out.println(list);
            listGammas.add(list);
        }
        br.close();

        double[][] gammas = new double[listGammas.size()][listGammas.get(0).size()];

        for (int i=0; i<gammas.length; i++) {
            for (int j=0; j<gammas[i].length; j++) {
                gammas[i][j] = listGammas.get(i).get(j);
            }
        }

        return gammas;
    }

    private static BMMClassifier loadModel(Config config, int i) throws Exception {
        String path = config.getString("output") + "/" + config.getString("modelName");
        File modelFile = new File(path, "iter." + i + ".model");
        BMMClassifier bmmClassifier = BMMClassifier.deserialize(modelFile);
        bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
        bmmClassifier.setPredictMode(config.getString("predictMode"));
        return bmmClassifier;
    }
}
