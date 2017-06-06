package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.util.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/9/14.
 */
public class AUC {
    public static double auc(Classifier.ProbabilityEstimator probEstimator, ClfDataSet dataSet){
        if (dataSet.getNumClasses()!=2){
            throw new IllegalArgumentException("dataSet.getNumClasses()!=2");
        }
        double[] probForOne = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> probEstimator.predictClassProbs(dataSet.getRow(i))[1])
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
        List<Pair<Double,Integer>> sortedPairs = IntStream.range(0,scores.length).parallel()
                .mapToObj(i -> new Pair<>(scores[i], labels[i]))
                .sorted(comparator.reversed()).collect(Collectors.toList());
        List<List<Double>> rates = getRates(sortedPairs);
        return area(rates);
    }

    /**
     * assume 1 is positive, 0 is negative
     * labels are sorted based on scores, from most likely to least likely of being positive
     * @param sortedPairs descending order by scores
     * @return
     */
    private static List<List<Double>> getRates(List<Pair<Double,Integer>> sortedPairs){
        int numData = sortedPairs.size();
        List<Double> truePositiveRates = new ArrayList<>();
        List<Double> falsePositiveRates = new ArrayList<>();
        double numPositives = sortedPairs.stream().parallel()
                .filter(pair -> pair.getSecond() == 1).count();
        double numNegatives = numData - numPositives;
        double truePositive = 0;
        double falsePositive = 0;
        truePositiveRates.add(0.0);
        falsePositiveRates.add(0.0);
        //start with all negative
        for (int i=0;i<numData;i++){
            Pair<Double, Integer> pair = sortedPairs.get(i);
            int label = pair.getSecond();
            double score = pair.getFirst();
            if (label ==1){
                truePositive += 1;
            } else {
                falsePositive += 1;
            }
            boolean condition1 = (i<numData-1)&&(score!=sortedPairs.get(i+1).getFirst());
            boolean condition2 = (i==numData-1);
            boolean condition = condition1||condition2;
            if (condition){
                truePositiveRates.add(truePositive/numPositives);
                falsePositiveRates.add(falsePositive/numNegatives);
            }
        }
        List<List<Double>> rates = new ArrayList<>();
        rates.add(truePositiveRates);
        rates.add(falsePositiveRates);
        return rates;
    }

    private static double area(List<List<Double>> rates){
        List<Double> tpr = rates.get(0);
        List<Double> fpr = rates.get(1);
//        System.out.println("true positive rates:");
//        System.out.println(tpr);
//        System.out.println("false positive rates:");
//        System.out.println(fpr);
        double tmp = IntStream.range(0,tpr.size()-1).parallel()
                .mapToDouble(i ->
                        (fpr.get(i) - fpr.get(i + 1)) * (tpr.get(i) + tpr.get(i + 1))).sum();
        double area = Math.abs(tmp)/2;
        return area;
    }
}
