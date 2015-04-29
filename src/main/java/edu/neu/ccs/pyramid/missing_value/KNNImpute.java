package edu.neu.ccs.pyramid.missing_value;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Created by chengli on 2/17/15.
 */
public class KNNImpute {
    private static final int N = 10;
    private static double[][] stats = null;

    /*
     * Normalize data if normFlag == 1
     * Save data as trec fomat
     */
    static ClfDataSet saveData(ClfDataSet data, List<String> featureNames, int normFlag)throws Exception{
        if (normFlag == 1) {
            if (stats == null) {
                stats = new double[data.getNumFeatures()][2];
            }
            for (int i = 0; i < data.getNumFeatures(); i++) {
                if (stats == null) {
                    for (int j = 0; j < data.getColumn(i).size(); j++) {
                        stats[i][0] += data.getColumn(i).get(j);
                    }
                    stats[i][0] = stats[i][0] * 1.0 / data.getColumn(i).size();
                    for (int j = 0; j < data.getColumn(i).size(); j++) {
                        stats[i][1] += Math.pow((data.getColumn(i).get(j) - stats[i][0]), 2);
                    }
                    stats[i][1] = Math.sqrt(stats[i][1]);
                }
                if (stats[i][1] == 0) {
                    continue;
                }
                for (int j = 0; j < data.getNumDataPoints(); j++) {
                    data.setFeatureValue(j, i, (data.getColumn(i).get(j) - stats[i][0]) * 1.0 / stats[i][1]);
                }
            }
        }

        DataSetUtil.setFeatureNames(data, featureNames);
        String[] extLabels = {"0","1"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        data.setLabelTranslator(labelTranslator);
        return data;
    }

    /*
     * Compute normalized Euclidean distance
     */
    public static double euclidean(Vector v1, Vector v2, int i) {
        double distance = 0;
        int num = 0;
        for (int j = 0; j < v1.size(); j++) {
            if (i != j && !Double.isNaN(v1.get(j)) && !Double.isNaN(v2.get(j))) {
                distance += Math.pow((v1.get(j) - v2.get(j)), 2.0);
                num++;
            }
        }
        return Math.sqrt(distance) / num;
    }

    /*
     * Input dataSet to be imputed, completeDataSet to calculate rmse, and trainDataSet to compute distance
     * Output imputed dataSet
     */
    public ClfDataSet impute(ClfDataSet dataSet, ClfDataSet completeDataSet, ClfDataSet trainDataSet) throws Exception{
        double rms = 0;
        double total = 0;
        ArrayList<Double> new_values = new ArrayList<Double>();
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            for (int j = 0; j < dataSet.getNumFeatures(); j++) {
                if (Double.isNaN(dataSet.getRow(i).get(j))) {
                    ArrayList<Double> distance = new ArrayList<Double>();
                    ArrayList<Double> value = new ArrayList<Double>();
                    for (int k = 0; k < trainDataSet.getNumDataPoints(); k++) {
                        if (i != k && !Double.isNaN(trainDataSet.getRow(k).get(j))) {
                            distance.add(euclidean(dataSet.getRow(i), trainDataSet.getRow(k), j));
                            value.add(trainDataSet.getRow(k).get(j));
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
                    dataSet.setFeatureValue(i, j, new_values.get(k));
                    k++;
                    rms += Math.pow(dataSet.getRow(i).get(j) -
                            completeDataSet.getRow(i).get(j), 2);
                }
                total += dataSet.getRow(i).get(j);
            }
        }
        double rmse = Math.sqrt(rms / k) / (total / (dataSet.getNumDataPoints() * dataSet.getNumFeatures()));
        System.out.println("RMSE: " + rmse);
        return dataSet;
    }
}