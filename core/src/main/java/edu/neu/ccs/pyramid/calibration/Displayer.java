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
    public static String displayCalibrationResult(Stream<Pair<Double, Double>> stream){
        final int numBuckets = 10;
        BucketInfo total = BucketInfo.aggregate(stream, numBuckets,0,1);
        double[] counts = total.getCounts();
        double[] correct = total.getSumLabels();
        double[] sumProbs = total.getSumProbs();
        double[] accs = new double[counts.length];
        double[] average_confidence = new double[counts.length];

        double datasetSize = 0;

        for (int i = 0; i < counts.length; i++) {
            accs[i] = correct[i] / counts[i];
            datasetSize += counts[i];
        }
        for (int j = 0; j < counts.length; j++) {
            average_confidence[j] = sumProbs[j] / counts[j];
        }

        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        DecimalFormat df = new DecimalFormat("#0.0%");
        StringBuilder sb = new StringBuilder();
        sb.append("interval\t\t").append("total\t\t\t").append("correct\t\t").append("incorrect\t").append("accuracy\t").append("average confidence\n");
        for (int i = 0; i < 10; i++) {
            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
                    .append("\t\t").append(counts[i]).append(" (").append(df.format((counts[i]/datasetSize))).append(")").append("\t\t").append(correct[i]).append("\t\t")
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






    public static String displayCalibrationResult(Stream<Pair<Double, Double>> stream, int numBuckets){
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
