package edu.neu.ccs.pyramid.missing_value;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by chengli on 2/17/15.
 */
public class KNNImpute {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    private static final int N = 10;

    public static void loadDataSet() {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(new File(DATASETS, "gene_expression/array.txt")));
            ArrayList<Integer> missing_row = new ArrayList<Integer>();
            ArrayList<Integer> missing_col = new ArrayList<Integer>();
            String line = reader.readLine();
            int feature_num = line.split("\t").length - 1;
            int row = 0;
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\t");
                for (int k = 2; k < words.length; k++) {
                    if (words[k].equals("")) {
                        if (!missing_row.contains(row)) {
                            missing_row.add(row);
                        }
                        if (!missing_col.contains(k)) {
                            missing_col.add(k);
                        }
                    }
                }
                row++;
            }
            reader.close();
            reader = new BufferedReader(new FileReader(new File(DATASETS, "gene_expression/array.txt")));
            BufferedWriter writer0 = new BufferedWriter(new FileWriter(new File(TMP, "gene_expression/feature_names.txt")));
            BufferedWriter writer1 = new BufferedWriter(new FileWriter(new File(TMP, "gene_expression/train_data.txt")));
            BufferedWriter writer2 = new BufferedWriter(new FileWriter(new File(TMP, "gene_expression/train_label.txt")));
            BufferedWriter writer3 = new BufferedWriter(new FileWriter(new File(TMP, "gene_expression/log_train_data.txt")));
            line = reader.readLine();
            String[] words = line.split("\t");
            for (int i = 4; i < words.length; i++) {
                if (!words[i].equals("")) {
                    writer0.write(words[i] + ":" + "\t" + "continuous." + "\n");
                }
            }
            writer0.close();
            int i = 0;
            while ((line = reader.readLine()) != null) {
                if (!missing_row.contains(i)) {
                    words = line.split("\t");
                    for (int k = 2; k < feature_num - 2; k++) {
                        if (!missing_col.contains(k)) {
                            writer1.write(words[k] + ",");
                            writer3.write(Double.toString(Math.log(Double.parseDouble(words[k]) + 1)) + ",");
                        }
                    }
                    writer1.write(words[feature_num - 2] + "\n");
                    writer3.write(Double.toString(Math.log(Double.parseDouble(words[feature_num - 2]) + 1)) + "\n");
                    if (words.length == feature_num) {
                        writer2.write(words[feature_num - 1] + "\n");
                    }
                    else {
                        writer2.write("0\n");
                    }
                }
                i++;
            }
            reader.close();
            writer1.close();
            writer2.close();
            writer3.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
    }

    static void saveTrainData()throws Exception{
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(6, new File(DATASETS, "gene_expression/log_train_data.txt"),
                new File(DATASETS, "gene_expression/train_label.txt"), ",", DataSetType.CLF_DENSE, false);

        DataSetUtil.setFeatureNames(data, featureNames);
        String[] extLabels = {"0","1","2","3","4","5"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        data.setLabelTranslator(labelTranslator);
        TRECFormat.save(data, new File(TMP, "gene_expression/train.trec"));
    }

    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<String>();
        try(BufferedReader br = new BufferedReader(new FileReader(new File(DATASETS, "gene_expression/feature_names.txt")))
        ){
            String line;
            while((line = br.readLine())!=null){
                String name = line.split(Pattern.quote(":"))[0];
                names.add(name);
            }
        }
        return names;
    }

    private static void produce_train(double p) throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "gene_expression/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        DataSetUtil.allowMissingValue(dataSet);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                if (Math.random()<p){
                    //todo change back
//                    dataSet.setFeatureValue(i,j,0);
                    dataSet.setFeatureValue(i,j,Double.NaN);
                }
            }
        }
        File folder = new File(TMP,"gene_expression/missing_value/"+p+"_missing");

        TRECFormat.save(dataSet,new File(folder,"train.trec"));
    }

    public static double euclidean(Vector v1, Vector v2, int i) {
        double distance = 0;
        for (int j = 0; j < v1.size(); j++) {
            if (i != j && !Double.isNaN(v1.get(j)) && !Double.isNaN(v2.get(j))) {
                distance += Math.pow((v1.get(j) - v2.get(j)), 2.0);
            }
        }
        return Math.sqrt(distance);
    }

    public static void impute(ClfDataSet dataSet) throws Exception{
        ClfDataSet completeDataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "gene_expression/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        double rms = 0;
        double total = 0;
        ArrayList<Double> new_values = new ArrayList<Double>();
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            for (int j = 0; j < dataSet.getNumFeatures(); j++) {
                if (Double.isNaN(dataSet.getRow(i).get(j))) {
                    ArrayList<Double> distance = new ArrayList<Double>();
                    ArrayList<Double> value = new ArrayList<Double>();
                    for (int k = 0; k < dataSet.getNumDataPoints(); k++) {
                        if (i != k && !Double.isNaN(dataSet.getRow(k).get(j))) {
                            distance.add(euclidean(dataSet.getRow(i), dataSet.getRow(k), j));
                            value.add(dataSet.getRow(k).get(j));
                        }
                    }
                    Integer[] indexes = new Integer[distance.size()];
                    for (int k = 0; k < indexes.length; k++) {
                        indexes[k] = k;
                    }
                    Arrays.sort(indexes, new Comparator<Integer>() {
                        @Override
                        public int compare(final Integer i1, final Integer i2) {
                            return Double.compare(distance.get(i1), distance.get(i2));
                        }
                    });
                    double final_value = 0;
                    double sum = 0;
                    for (int k = 0; k < N; k++) {
                        final_value += (1.0 / (1+distance.get(indexes[k]))) * value.get(indexes[k]);
                        sum += (1.0 / (1+distance.get(indexes[k])));
                    }
                    new_values.add(final_value / sum);
                }
            }
            System.out.println(i);
        }
        int k = 0;
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            for (int j = 0; j < dataSet.getNumFeatures(); j++) {
                if (Double.isNaN(dataSet.getRow(i).get(j))) {
                    dataSet.getRow(i).set(j, new_values.get(k));
                    dataSet.getColumn(j).set(i, new_values.get(k));
                    k++;
                    rms += Math.pow(dataSet.getRow(i).get(j) - completeDataSet.getRow(i).get(j), 2);
                }
                total += dataSet.getRow(i).get(j);
            }
        }
        double rmse = Math.sqrt(rms / k) / (total / (dataSet.getNumDataPoints() * dataSet.getNumFeatures()));
        System.out.println("RMSE: " + rmse);
    }
    public static void main(String[] args) throws Exception{
//        loadDataSet();
//        saveTrainData();
//        double[] percentages = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
//        for (double p: percentages){
//            produce_train(p);
//        }
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/gene_expression/missing_value/0.2_missing/train.trec"),
                DataSetType.CLF_SPARSE, false);
        impute(dataSet);
        TRECFormat.save(dataSet, new File(TMP, "/gene_expression/impute/0.2_missing/train.trec"));
        System.out.println();
    }
}