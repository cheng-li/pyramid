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
        int total = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            total += bucketInfo.getCounts()[i];
        }
        return sum/total;
    }

    public static double sharpness(Stream<Pair<Double, Integer>> stream, int numBuckets){

        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets);
        double[] accuracies = new double[bucketInfo.getNumBuckets()];
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            accuracies[i] = bucketInfo.getSums()[i]/bucketInfo.getCounts()[i];
        }
        double average = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            average += bucketInfo.getSums()[i];
        }

        int total = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            total += bucketInfo.getCounts()[i];
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
        VarResult varResult = stream.collect(()->new VarResult(),VarResult::add, VarResult::add);
        double count = varResult.count;
        double sum = varResult.sum;
        double sumSquare = varResult.sumSquare;
        return (sumSquare-sum*sum/count)/(count-1);

    }



    public static class VarResult{
        double sum;
        double sumSquare;
        double count;

        public VarResult() {
        }

         void add(VarResult varResult){
            this.sum += varResult.sum;
            this.count += varResult.count;
            this.sumSquare += varResult.sumSquare;
        }

         void add(Pair<Double, Integer> pair){
            this.sum += pair.getSecond();
            this.count += 1;
            this.sumSquare += Math.pow(pair.getSecond(),2);
        }

    }
}
