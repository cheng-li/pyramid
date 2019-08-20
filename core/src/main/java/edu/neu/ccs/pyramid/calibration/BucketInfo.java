package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BucketInfo {
    public double[] counts;
    public double[] sumLabels;
    public double [] sumProbs;
    public double[] sumSquareProbs;
    public int numBuckets;
    public double minValue;
    public double maxValue;

    private BucketInfo(int size, double minValue, double maxValue) {
        numBuckets = size;
        counts = new double[size];
        sumLabels = new double[size];
        sumProbs = new double[size];
        sumSquareProbs = new double[size];
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

//
//    private BucketInfo(double[] counts, double[] sumLabels, double[] sumProbs, double[] sumSquareProbs) {
//        this.numBuckets = counts.length;
//        this.counts = counts;
//        this.sumLabels = sumLabels;
//        this.sumProbs = sumProbs;
//        this.sumSquareProbs = sumSquareProbs;
//    }


    public static BucketInfo aggregate(Stream<Pair<Double,Double>> stream, int numBuckets, double minValue, double maxValue){
        return stream.collect(()->new BucketInfo(numBuckets, minValue, maxValue),BucketInfo::add, BucketInfo::addAll);
    }

    public double[] getCounts() {
        return counts;
    }

    public double[] getSumLabels() {
        return sumLabels;
    }

    public double[] getSumProbs() {
        return sumProbs;
    }

    public double[] getSumSquareProbs() {
        return sumSquareProbs;
    }

    public int getNumBuckets() {
        return numBuckets;
    }


    public double[] getAveLabels(){
        double[] ave = new double[numBuckets];
        for (int i=0;i<numBuckets;i++){
            ave[i] = sumLabels[i]/counts[i];
        }
        return ave;
    }

    public double[] getAveProbs(){
        double[] ave = new double[numBuckets];
        for (int i=0;i<numBuckets;i++){
            ave[i] = sumProbs[i]/counts[i];
        }
        return ave;
    }

    public double getTotalCount(){
        return MathUtil.arraySum(counts);
    }


    public void addAll(BucketInfo bucketInfo2){
        for (int i=0;i<this.counts.length;i++){
            this.counts[i] += bucketInfo2.counts[i];
            this.sumLabels[i] += bucketInfo2.sumLabels[i];
            this.sumProbs[i] += bucketInfo2.sumProbs[i];
            this.sumSquareProbs[i] += bucketInfo2.sumSquareProbs[i];
        }
    }


    public void add(Pair<Double,Double> pair){
        final int numBuckets = this.counts.length;
        double bucketLength = (maxValue-minValue)/numBuckets;
        double prob = pair.getFirst();
        int index = (int)Math.floor((prob-minValue)/bucketLength);
        if (index<0){
            index=0;
        }
        if (index>=numBuckets){
            index = numBuckets-1;
        }
        this.counts[index] += 1;
        this.sumProbs[index] += prob;
        this.sumLabels[index]+=pair.getSecond();
        this.sumSquareProbs[index] += prob*prob;
    }

//    public static BucketInfo addTo(BucketInfo bucketInfo, Pair<Double,Integer> pair){
//        final int numBuckets = bucketInfo.counts.length;
//        double bucketLength = 1.0/numBuckets;
//        double prob = pair.getFirst();
//        int index = (int)Math.floor(prob/bucketLength);
//        if (index<0){
//            index=0;
//        }
//        if (index>=numBuckets){
//            index = numBuckets-1;
//        }
//        bucketInfo.counts[index] += 1;
//        bucketInfo.sumProbs[index] += prob;
//        bucketInfo.sums[index]+=pair.getSecond();
//        return bucketInfo;
//    }


//    public static BucketInfo addTo(BucketInfo bucketInfo, Pair<Double,Integer> pair){
//        BucketInfo res = new BucketInfo(bucketInfo.counts.length);
//        res.sums = Arrays.copyOf(bucketInfo.sums,bucketInfo.sums.length);
//        res.sumProbs = Arrays.copyOf(bucketInfo.sumProbs, bucketInfo.sumProbs.length);
//        res.counts = Arrays.copyOf(bucketInfo.counts, bucketInfo.counts.length);
//        final int numBuckets = bucketInfo.counts.length;
//        double bucketLength = 1.0/numBuckets;
//        double prob = pair.getFirst();
//        int index = (int)Math.floor(prob/bucketLength);
//        if (index<0){
//            index=0;
//        }
//        if (index>=numBuckets){
//            index = numBuckets-1;
//        }
//        res.counts[index] += 1;
//        res.sumProbs[index] += prob;
//        res.sums[index]+=pair.getSecond();
//        return res;
//    }

    @Override
    public String toString() {
        return "BucketInfo{" +
                "counts=" + Arrays.toString(counts) +
                ", sumLabels=" + Arrays.toString(sumLabels) +
                ", sumProbs=" + Arrays.toString(sumProbs) +
                '}';
    }
}
