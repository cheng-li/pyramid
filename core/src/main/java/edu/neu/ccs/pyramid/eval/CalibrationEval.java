package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.calibration.BucketInfo;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.stream.Stream;

public class CalibrationEval {
    
    public static double mse(Stream<Pair<Double, Integer>> stream){
        return stream.mapToDouble(pair->Math.pow(pair.getFirst()-pair.getSecond(),2)).average().getAsDouble();
    }

    public static double error(Stream<Pair<Double, Integer>> stream, int numBuckets){
        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets);
        double sum = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            double count = bucketInfo.getCounts()[i];
            sum += count*Math.abs(bucketInfo.getSumProbs()[i]/count-bucketInfo.getSums()[i]/count);
        }
        return sum/stream.count();
    }

    public static double sharpness(Stream<Pair<Double, Integer>> stream, int numBuckets){
        long total = stream.count();
        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets);
        double[] accuracies = new double[bucketInfo.getNumBuckets()];
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            accuracies[i] = bucketInfo.getSums()[i]/bucketInfo.getCounts()[i];
        }
        double average = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            average += bucketInfo.getSums()[i];
        }

        average/=total;
        double sum = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            double count = bucketInfo.getCounts()[i];
            sum += count*Math.pow(accuracies[i]-average,2);
        }
        return sum/(total-1);
    }

    public static double variance(Stream<Pair<Double, Integer>> stream){
        long count = stream.count();
        double sum = stream.mapToDouble(pair->pair.getSecond()).sum();
        double sumSquare = stream.mapToDouble(pair->Math.pow(pair.getSecond(),2)).sum();
        return (sumSquare-sum*sum/count)/(count-1);

    }


}
