package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Confidence Threshold for Target Accuracy
 */
public class CTAT {

    /**
     * Find confidence threshold for target accuracy on validation dataset
     * @param confidence prediction confidence value for each instance in validation set
     * @param correctness correctness (true/false) of the prediction for each instance in validation set
     * @param targetAccuracy target accuracy level
     * @return summary that contains threshold, and various
     *          auto-coding performance metrics on the validation set such as auto-coding percentage, auto-coding accuracy,
     *          number of automated instances, number of correctly automated instances
     */
    public static Summary findThreshold(double[] confidence, boolean[] correctness, double targetAccuracy){
        if (confidence.length!=correctness.length){
            throw new IllegalArgumentException("confidence.length!=correctness.length");
        }
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, confidence.length).mapToObj(i->{
            Pair<Double,Integer> pair = new Pair<>(confidence[i],0);
            if (correctness[i]){
                pair.setSecond(1);
            }
            return pair;
        });
        return findThreshold(stream, targetAccuracy);
    }

    /**
     * Apply the given confidence threshold to test dataset and get auto-coding summary
     * @param confidence prediction confidence value for each instance in test set
     * @param correctness correctness (true/false) of the prediction for each instance in test set
     * @param confidenceThreshold given confidence threshold
     * @return summary that contains (given) threshold, and various
     *          auto-coding performance metrics on the test set such as auto-coding percentage, auto-coding accuracy,
     *          number of automated instances, number of correctly automated instances
     */
    public static Summary applyThreshold(double[] confidence, boolean[] correctness, double confidenceThreshold){
        if (confidence.length!=correctness.length){
            throw new IllegalArgumentException("confidence.length!=correctness.length");
        }
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, confidence.length).mapToObj(i->{
            Pair<Double,Integer> pair = new Pair<>(confidence[i],0);
            if (correctness[i]){
                pair.setSecond(1);
            }
            return pair;
        });
        return applyThreshold(stream,confidenceThreshold);
    }


    public static double clip(double original, double lowerBound, double upperBound){
        if (original>upperBound){
            return upperBound;
        }

        if (original<lowerBound){
            return lowerBound;
        }

        return original;
    }

    public static Summary findThreshold(Stream<Pair<Double, Integer>> stream, double targetAccuracy){
        Comparator<Pair<Double, Integer>> comparator = Comparator.comparing(Pair::getFirst);
        List<Pair<Double,Integer>> list = stream.sorted(comparator.reversed()).collect(Collectors.toList());
        Summary summary = new Summary();
        int numCorrect = 0;
        int size = list.size();
        Pair<Double, Double> result =new Pair<>();
        for (int i = 0; i < size; i++){
            numCorrect += list.get(i).getSecond();
            double accuracy = (numCorrect*1.0)/(i+1);
            if (i==size-1||(i<size-1&&(!list.get(i).getFirst().equals(list.get(i+1).getFirst())))){
                if (accuracy >= targetAccuracy ){
                    summary.confidenceThreshold = list.get(i).getFirst();
                    summary.autoCodingPercentage = (i+1)/(size*1.0);
                    summary.autoCodingAccuracy = accuracy;
                    summary.numAutoCoded = (i+1);
                    summary.numCorrectAutoCoded = numCorrect;
                }
            }

        }
        return summary;
    }


    public static Summary applyThreshold(Stream<Pair<Double, Integer>> stream, double confidenceThreshold){
        List<Pair<Double,Integer>> list = stream.collect(Collectors.toList());
        int sum = 0;
        int correct = 0;
        int size = list.size();
        for (int i = 0; i<size; i++){
            if (list.get(i).getFirst() >= confidenceThreshold){
                sum++;
                if(list.get(i).getSecond() == 1){
                    correct += 1;
                }
            }
        }
        Summary summary = new Summary();
        summary.confidenceThreshold=confidenceThreshold;
        summary.autoCodingPercentage = (sum*1.0)/size;
        summary.autoCodingAccuracy = (correct*1.0)/sum;
        summary.numAutoCoded = sum;
        summary.numCorrectAutoCoded = correct;
        return summary;
    }

    public static class AllThresholdResult{
        public List<Double> thresholds;
        public List<Double>  accuracies;
        public List<Double> percentages;
        public List<Double> interpolatedAccuracies;

        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("confidence").append("\t").append("percentage").append("\t").append("accuracy").append("\t").append("interpolated accuracy").append("\n");
            for (int i=0;i<thresholds.size();i++){
                stringBuilder.append(thresholds.get(i)).append("\t").append(percentages.get(i)).append("\t").append(accuracies.get(i)).append("\t").append(interpolatedAccuracies.get(i)).append("\n");
            }
            return stringBuilder.toString();
        }
    }


    /**
     *
     * @param predictionResults pair of (confidence, correctness)
     * @return
     */
    public static AllThresholdResult showAllThresholds(List<Pair<Double,Double>> predictionResults){
        Comparator<Pair<Double,Double>> comparator = Comparator.comparing(Pair::getFirst);
        List<Pair<Double,Double>> sorted = predictionResults.stream().sorted(comparator.reversed()).collect(Collectors.toList());


        double total = 0;
        double correct = 0;
        List<Double> thresholds = new ArrayList<>();
        List<Double> accuracies = new ArrayList<>();
        List<Double> fractions = new ArrayList<>();
        for (int i=0;i<sorted.size();i++){
            Pair<Double,Double> pair = sorted.get(i);
            total+=1;
            correct += pair.getSecond();

            double current = pair.getFirst();
            if ((i<sorted.size()-1&&!sorted.get(i).getFirst().equals(sorted.get(i+1).getFirst()))||i==sorted.size()-1){
                thresholds.add(current);
                accuracies.add(correct*1.0/total);
                fractions.add(total/predictionResults.size());
            }
        }

        Collections.reverse(thresholds);
        Collections.reverse(accuracies);
        Collections.reverse(fractions);


        List<Double> interpolatedAccuracies = interpolatedAccuracies(accuracies);

        AllThresholdResult allThresholdResult = new AllThresholdResult();
        allThresholdResult.accuracies = accuracies;
        allThresholdResult.thresholds = thresholds;
        allThresholdResult.interpolatedAccuracies = interpolatedAccuracies;
        allThresholdResult.percentages=fractions;
        return allThresholdResult;
    }


    private static List<Double>  interpolatedAccuracies(List<Double> accuracies){
        double max = 0;
        List<Double> inter = new ArrayList<>();
        for (double acc: accuracies){
            max = Math.max(max,acc);
            inter.add(max);
        }
        return inter;
    }


    public static class Summary {
        double confidenceThreshold=1.1;
        double autoCodingPercentage =0;
        double autoCodingAccuracy =Double.NaN;
        int numAutoCoded =0;
        int numCorrectAutoCoded =0;


        public double getConfidenceThreshold() {
            return confidenceThreshold;
        }

        public double getAutoCodingPercentage() {
            return autoCodingPercentage;
        }

        public double getAutoCodingAccuracy() {
            return autoCodingAccuracy;
        }

        public int getNumAutoCoded() {
            return numAutoCoded;
        }

        public int getNumCorrectAutoCoded() {
            return numCorrectAutoCoded;
        }

        @Override
        public String toString() {
            return "Summary{" +
                    "confidenceThreshold=" + confidenceThreshold +
                    ", autoCodingPercentage=" + autoCodingPercentage +
                    ", autoCodingAccuracy=" + autoCodingAccuracy +
                    ", numAutoCoded=" + numAutoCoded +
                    ", numCorrectAutoCoded=" + numCorrectAutoCoded +
                    '}';
        }
    }

}
