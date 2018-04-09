package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.text.DecimalFormat;
import java.util.stream.Stream;

public class Displayer {

    /**
     *
     * @param stream containing calibrated confidence
     * @return
     */
    public static String displayCalibrationResult(Stream<Pair<Double, Integer>> stream){
        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;
        BucketInfo total;

        total = stream.map(doubleIntegerPair -> {
            double probs = doubleIntegerPair.getFirst();
            double[] sum = new double[numBuckets];
            double[] sumProbs = new double[numBuckets];
            double[] count = new double[numBuckets];
            int index = (int)Math.floor(probs/bucketLength);
            if (index<0){
                index=0;
            }
            if (index>=numBuckets){
                index = numBuckets-1;
            }
            count[index] += 1;
            sumProbs[index] += probs;
            sum[index]+=doubleIntegerPair.getSecond();
            return new BucketInfo(count, sum,sumProbs);
        }).collect(() -> new BucketInfo(numBuckets), BucketInfo::addAll, BucketInfo::addAll);

        double[] counts = total.getCounts();
        double[] correct = total.getSums();
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
                    .append(counts[i] - correct[i]).append("\t\t").append(decimalFormat.format(accs[i])).append("\t\t")
                    .append(decimalFormat.format(average_confidence[i])).append("\n");

        }

        String result = sb.toString();
        return result;

    }
}
