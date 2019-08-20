package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.calibration.BucketInfo;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.stream.Stream;

public class CalibrationEval {
    
    public static double mse(Stream<Pair<Double, Double>> stream){
        return stream.mapToDouble(pair->Math.pow(pair.getFirst()-pair.getSecond(),2)).average().getAsDouble();
    }

    public static double absoluteError(Stream<Pair<Double, Double>> stream, int numBuckets){
        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets,0,1);
        double sum = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            sum += Math.abs(bucketInfo.getSumProbs()[i]-bucketInfo.getSumLabels()[i]);
        }

        return sum/bucketInfo.getTotalCount();
    }

    public static double squareError(Stream<Pair<Double, Double>> stream, int numBuckets){
        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets,0,1);
        double[] aveLabels = bucketInfo.getAveLabels();
        double sum = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            if (bucketInfo.getCounts()[i]!=0){
                sum += Math.pow(aveLabels[i],2)*bucketInfo.getCounts()[i]-2*bucketInfo.getSumProbs()[i]*aveLabels[i]+bucketInfo.getSumSquareProbs()[i];
            }

        }
        return sum/bucketInfo.getTotalCount();
    }

    public static double sharpness(Stream<Pair<Double, Double>> stream, int numBuckets){

        BucketInfo bucketInfo = BucketInfo.aggregate(stream, numBuckets,0,1);
        double[] accuracies = new double[bucketInfo.getNumBuckets()];
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            accuracies[i] = bucketInfo.getSumLabels()[i]/bucketInfo.getCounts()[i];
        }
        double average = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            average += bucketInfo.getSumLabels()[i];
        }

        double total = bucketInfo.getTotalCount();
        average/=total;
        double sum = 0;
        for (int i=0;i<bucketInfo.getNumBuckets();i++){
            double count = bucketInfo.getCounts()[i];
            if (count!=0){
                sum += count*Math.pow(accuracies[i]-average,2);
            }

        }
        return sum/(total-1);
    }

    public static double variance(Stream<Pair<Double, Double>> stream){
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

         void add(Pair<Double, Double> pair){
            this.sum += pair.getSecond();
            this.count += 1;
            this.sumSquare += Math.pow(pair.getSecond(),2);
        }

    }
}
