package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

public class BucketInfo {

    public BucketInfo(int size) {
        counts = new double[size];
        sums = new double[size];
        sumProbs = new double[size];
    }

    public BucketInfo(double[] counts, double[] sums, double[] sumProbs) {
        this.counts = counts;
        this.sums = sums;
        this.sumProbs = sumProbs;
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

    double[] counts;
    double[] sums;
    double [] sumProbs;

    public static BucketInfo add(BucketInfo bucketInfo1, BucketInfo bucketInfo2){
        BucketInfo bucketInfo = new BucketInfo(bucketInfo1.counts.length);
        for (int i=0;i<bucketInfo1.counts.length;i++){
            bucketInfo.counts[i] = bucketInfo1.counts[i]+bucketInfo2.counts[i];
            bucketInfo.sums[i] = bucketInfo1.sums[i]+bucketInfo2.sums[i];
            bucketInfo.sumProbs[i] = bucketInfo1.sumProbs[i] + bucketInfo2.sumProbs[i];
        }
        return bucketInfo;
    }
}
