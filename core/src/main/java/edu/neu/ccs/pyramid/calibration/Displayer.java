package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.text.DecimalFormat;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Displayer {

    /**
     *
     * @param stream containing calibrated confidence
     * @return
     */
    public static String displayCalibrationResult(Stream<Pair<Double, Integer>> stream){
        final int numBuckets = 10;
        BucketInfo total = BucketInfo.aggregate(stream, numBuckets,0,1);
        double[] counts = total.getCounts();
        double[] correct = total.getSumLabels();
        double[] sumProbs = total.getSumProbs();
        double[] accs = new double[counts.length];
        double[] average_confidence = new double[counts.length];

        for (int i = 0; i < counts.length; i++) {
            accs[i] = correct[i] / counts[i];
        }
        for (int j = 0; j < counts.length; j++) {
            average_confidence[j] = sumProbs[j] / counts[j];
        }

        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
        for (int i = 0; i < 10; i++) {
            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
                    .append(counts[i] - correct[i]).append("\t\t");
                    if (Double.isFinite(accs[i])){
                        sb.append(decimalFormat.format(accs[i])).append("\t\t");
                    } else {
                        sb.append("N/A").append("\t\t");
                    }

                    if (Double.isFinite(average_confidence[i])){
                        sb.append(decimalFormat.format(average_confidence[i])).append("\n");
                    } else {
                        sb.append("N/A").append("\n");
                    }

        }

        String result = sb.toString();
        return result;

    }


    public static Pair<Double, Double> confidenceThresholdForAccuraccyTarget(double targetaccuracy,Stream<Pair<Double, Integer>> stream){
        Comparator<Pair<Double, Integer>> comparator = Comparator.comparing(pair->pair.getFirst());
        List<Pair<Double,Integer>> list = stream.sorted(comparator.reversed()).collect(Collectors.toList());
        int sumCorrect = 0;
        double confidenceThreshold = 0;
        double autocodePercentage = 0;
        int size = list.size();
        Pair<Double, Double> result =new Pair<>();
        for (int i = 0; i < size; i++){
            sumCorrect += list.get(i).getSecond();
            double current_accuracy = (sumCorrect*1.0)/(i+1);
            if (current_accuracy >= targetaccuracy ){
                confidenceThreshold = list.get(i).getFirst();
                autocodePercentage = (i+1)/(size*1.0);
            }
        }
        result.setFirst(confidenceThreshold);
        result.setSecond(autocodePercentage);
        return result;
    }



    public static String displayCalibrationResult(Stream<Pair<Double, Integer>> stream, int numBuckets){
        BucketInfo total = BucketInfo.aggregate(stream, numBuckets,0,1);
        double[] counts = total.getCounts();
        double[] correct = total.getSumLabels();
        double[] sumProbs = total.getSumProbs();
        double[] accs = new double[counts.length];
        double[] average_confidence = new double[counts.length];

        for (int i = 0; i < counts.length; i++) {
            accs[i] = correct[i] / counts[i];
        }
        for (int j = 0; j < counts.length; j++) {
            average_confidence[j] = sumProbs[j] / counts[j];
        }

        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
        for (int i = 0; i < numBuckets; i++) {
            sb.append("[").append(decimalFormat.format(i * 1.0/numBuckets)).append(",")
                    .append(decimalFormat.format((i + 1) * 1.0/numBuckets)).append("]")
                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
                    .append(counts[i] - correct[i]).append("\t\t");
            if (Double.isFinite(accs[i])){
                sb.append(decimalFormat.format(accs[i])).append("\t\t");
            } else {
                sb.append("N/A").append("\t\t");
            }

            if (Double.isFinite(average_confidence[i])){
                sb.append(decimalFormat.format(average_confidence[i])).append("\n");
            } else {
                sb.append("N/A").append("\n");
            }

        }

        String result = sb.toString();
        return result;

    }
}
