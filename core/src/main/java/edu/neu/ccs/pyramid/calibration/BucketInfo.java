package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.Arrays;
import java.util.stream.Stream;

public class BucketInfo {
    public double[] counts;
    public double[] sums;
    public double [] sumProbs;
    public int numBuckets;

    public BucketInfo(int size) {
        numBuckets = size;
        counts = new double[size];
        sums = new double[size];
        sumProbs = new double[size];
    }


    public BucketInfo(double[] counts, double[] sums, double[] sumProbs) {
        this.numBuckets = counts.length;
        this.counts = counts;
        this.sums = sums;
        this.sumProbs = sumProbs;
    }


    public static BucketInfo aggregate(Stream<Pair<Double,Integer>> stream, int numBuckets){
        return stream.collect(()->new BucketInfo(numBuckets),BucketInfo::add, BucketInfo::addAll);
    }

    public double[] getCounts() {
        return counts;
    }

    public double[] getSums() {
        return sums;
    }

    public double[] getSumProbs() {
        return sumProbs;
    }

    public int getNumBuckets() {
        return numBuckets;
    }

    //    public static BucketInfo add(BucketInfo bucketInfo1, BucketInfo bucketInfo2){
//        BucketInfo bucketInfo = new BucketInfo(bucketInfo1.counts.length);
//        for (int i=0;i<bucketInfo1.counts.length;i++){
//            bucketInfo.counts[i] = bucketInfo1.counts[i]+bucketInfo2.counts[i];
//            bucketInfo.sums[i] = bucketInfo1.sums[i]+bucketInfo2.sums[i];
//            bucketInfo.sumProbs[i] = bucketInfo1.sumProbs[i] + bucketInfo2.sumProbs[i];
//        }
//        return bucketInfo;
//    }


    public void addAll(BucketInfo bucketInfo2){
        for (int i=0;i<this.counts.length;i++){
            this.counts[i] += bucketInfo2.counts[i];
            this.sums[i] += bucketInfo2.sums[i];
            this.sumProbs[i] += bucketInfo2.sumProbs[i];
        }
    }


    public void add(Pair<Double,Integer> pair){
        final int numBuckets = this.counts.length;
        double bucketLength = 1.0/numBuckets;
        double prob = pair.getFirst();
        int index = (int)Math.floor(prob/bucketLength);
        if (index<0){
            index=0;
        }
        if (index>=numBuckets){
            index = numBuckets-1;
        }
        this.counts[index] += 1;
        this.sumProbs[index] += prob;
        this.sums[index]+=pair.getSecond();
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
                ", sums=" + Arrays.toString(sums) +
                ", sumProbs=" + Arrays.toString(sumProbs) +
                '}';
    }
}
