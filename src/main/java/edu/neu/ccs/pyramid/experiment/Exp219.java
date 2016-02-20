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
            List<Double> gammasPerList = getPerplexityList(gammas);
            List<Double> PIsPerList = getPerplexityList(PIs);

            Map<Double, HashMap<Boolean, Double>> gammasPerplexityMap = getPerplexityMap(labels, predictions, gammasPerList);
            Map<Double, HashMap<Boolean, Double>> PIsPerplexityMap = getPerplexityMap(labels, predictions, PIsPerList);


            Map<Double, HashMap<Boolean, Double>> KLDivergence = getKL(labels, predictions, gammas, PIs, gammasPerList);

            writePerplexityFile(gammasPerplexityMap, path, i, "gammas");
            writePerplexityFile(PIsPerplexityMap, path, i, "PIs");
            writePerplexityFile(KLDivergence, path, i, "KL");
        }

    }

    private static List<Double> getPerplexityList(double[][] gammas) {
        List<Double> perplexityList = new ArrayList<>();
        for (int i=0; i<gammas.length; i++) {
            perplexityList.add(getPerplexity(gammas[i]));
        }
        return perplexityList;
    }

    private static double getPerplexity(double[] dist) {
        double entropy = 0.0;
        for (int j=0; j<dist.length; j++) {
            double p = dist[j];
            entropy -= p * Math.log(p)/Math.log(2);
        }
        return Math.pow(2, entropy);
    }

    private static Map<Double, HashMap<Boolean, Double>> getPerplexityMap(MultiLabel[] labels, MultiLabel[] predictions, List<Double> perplexities) {
        Map<Double, HashMap<Boolean, Double>> perplexityMap = new TreeMap<>();

        for (int i=0; i<perplexities.size(); i++) {
            double perplexity = perplexities.get(i);
            boolean isMatch = labels[i].equals(predictions[i]);
            if (!perplexityMap.containsKey(perplexity)) {
                perplexityMap.put(perplexity, new HashMap<>());
            }
            if (!perplexityMap.get(perplexity).containsKey(isMatch)) {
                perplexityMap.get(perplexity).put(isMatch, 1.0);
            } else {
                perplexityMap.get(perplexity).put(isMatch, perplexityMap.get(perplexity).get(isMatch)+1);
            }
        }

        return perplexityMap;
    }

    private static double getKL(double[] Ps, double[] Qs) {
        double kl = 0.0;
        for (int j=0; j<Ps.length; j++) {
            double p = Ps[j];
            double q = Qs[j];
            kl += p * Math.log(p/q)/Math.log(2);
        }
        return kl;
    }

    private static Map<Double,HashMap<Boolean,Double>> getKL(MultiLabel[] labels, MultiLabel[] predictions, double[][] gammas, double[][] PIs, List<Double> perplexities) {

        Map<Double, HashMap<Boolean, Double>> KLMap = new TreeMap<>();
        Map<Double, HashMap<Boolean, Double>> KLCountMap = new TreeMap<>();

        for (int i=0; i<perplexities.size(); i++) {
            double perplexity = (double) Math.round(perplexities.get(i));
            double kl = getKL(PIs[i], gammas[i]);
            boolean isMath = labels[i].equals(predictions[i]);
            if (!KLMap.containsKey(perplexity)) {
                KLMap.put(perplexity, new HashMap<>());
                KLCountMap.put(perplexity, new HashMap<>());
            }
            if (!KLMap.get(perplexity).containsKey(isMath)) {
                KLMap.get(perplexity).put(isMath, kl);
                KLCountMap.get(perplexity).put(isMath, 1.0);
            } else {
                KLMap.get(perplexity).put(isMath, KLMap.get(perplexity).get(isMath)+kl);
                KLCountMap.get(perplexity).put(isMath, KLCountMap.get(perplexity).get(isMath)+1);
            }
        }

        for (double perplexity : KLMap.keySet()) {
            for (boolean isMath : KLMap.get(perplexity).keySet()) {
                double count = KLCountMap.get(perplexity).get(isMath);
                double sumKL = KLMap.get(perplexity).get(isMath);
                double avgKL = sumKL / count;
                KLMap.get(perplexity).put(isMath, avgKL);
            }
        }
        return KLMap;
    }

    private static void writePerplexityFile(Map<Double, HashMap<Boolean,Double>> perplexityMap, String path,
                                            int iter, String type) throws IOException {
        File file = new File(path, "iter."+iter+"."+type+".perplexity");
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        for (Map.Entry<Double, HashMap<Boolean, Double>> entry : perplexityMap.entrySet()) {
            double perplexity = entry.getKey();
            HashMap<Boolean, Double> hMap = entry.getValue();
            double falseCount = hMap.containsKey(false) ? hMap.get(false) : 0.0;
            double trueCount = hMap.containsKey(true) ? hMap.get(true) : 0.0;

            bw.write(perplexity + "\t" + falseCount + "\t" + trueCount + "\n");
        }
        bw.close();
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
