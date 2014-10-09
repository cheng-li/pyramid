package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/9/14.
 */

//todo: there seems to be a bug, see HistogramNBTest
public class AUC {
    public static double auc(ProbabilityEstimator probEstimator, ClfDataSet dataSet){
        double[] probForOne = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> probEstimator.predictClassProbs(dataSet.getFeatureRow(i))[1])
                .toArray();
        int[] labels = dataSet.getLabels();
        return auc(probForOne,labels);
    }

    /**
     * original order, unsorted
     * @param scores
     * @param labels
     * @return
     */
    public static double auc(double[] scores, int[] labels){
        Comparator<Pair<Double,Integer>> comparator = Comparator.comparing(Pair::getFirst);
        int[] sortedLabels = IntStream.range(0,scores.length).parallel()
                .mapToObj(i -> new Pair<>(scores[i], labels[i]))
                .sorted(comparator.reversed()).mapToInt(Pair::getSecond).toArray();
        List<double[]> rates = getRates(sortedLabels);
        return area(rates);
    }

    /**
     * assume 1 is positive, 0 is negative
     * labels are sorted based on scores, from most likely to least likely of being positive
     * @param sortedLabels
     * @return
     */
    private static List<double[]> getRates(int[] sortedLabels){
        int numData = sortedLabels.length;
        double[] truePositiveRates = new double[numData+1];
        double[] falsePositiveRates = new double[numData+1];
        double numPositives = Arrays.stream(sortedLabels).filter(label -> label==1).count();
        double numNegatives = numData - numPositives;
        double truePositive = 0;
        double falsePositive = 0;
        truePositiveRates[0] = 0;
        falsePositiveRates[0] = 0;
        //start with all negative
        for (int i=1;i<=numData;i++){
            int label = sortedLabels[i-1];
            if (label ==1){
                truePositive += 1;
            } else {
                falsePositive += 1;
            }
            truePositiveRates[i] = truePositive/numPositives;
            falsePositiveRates[i] = falsePositive/numNegatives;
        }
        List<double[]> rates = new ArrayList<>();
        rates.add(truePositiveRates);
        rates.add(falsePositiveRates);
        return rates;
    }

    private static double area(List<double[]> rates){
        double[] tpr = rates.get(0);
        double[] fpr = rates.get(1);
        double tmp = IntStream.range(0,tpr.length-1).parallel()
                .mapToDouble(i ->
                        (fpr[i]-fpr[i+1])*(tpr[i]+tpr[i+1])).sum();
        double area = Math.abs(tmp)/2;
        return area;
    }
}
