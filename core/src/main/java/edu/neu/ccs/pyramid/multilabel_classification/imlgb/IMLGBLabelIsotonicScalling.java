package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.List;
import java.util.stream.IntStream;

public class IMLGBLabelIsotonicScalling implements Serializable {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private IMLGradientBoosting imlGradientBoosting;

    public IMLGBLabelIsotonicScalling(IMLGradientBoosting imlGradientBoosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        this.imlGradientBoosting = imlGradientBoosting;
        final int numBuckets = 10000;
        double bucketLength = 1.0/numBuckets;
        double[] locations = new double[numBuckets];
        for (int i=0;i<numBuckets;i++){
            locations[i]= i*bucketLength + 0.5*bucketLength;
        }

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    System.out.println(i);
                    double[] probs = imlGradientBoosting.predictClassProbs(multiLabelClfDataSet.getRow(i));
                    double[] count = new double[numBuckets];
                    double[] sum = new double[numBuckets];
                    double[] sumProbs = new double[numBuckets];
                    for (int a=0;a<probs.length;a++){
                        int index = (int)Math.floor(probs[a]/bucketLength);
                        if (index<0){
                            index=0;
                        }
                        if (index>=numBuckets){
                            index = numBuckets-1;
                        }
                        count[index] += 1;
                        sumProbs[index] += probs[a];
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(a)){
                            sum[index] += 1;
                        } else {
                            sum[index] += 0;
                        }
                    }
                    return new BucketInfo(count, sum, sumProbs);
                }).reduce(empty, BucketInfo::add, BucketInfo::add);
        double[] counts = total.counts;
        double[] sums = total.sums;
        double[] accs = new double[counts.length];
        for (int i=0;i<counts.length;i++){
            if (counts[i]!=0){
                accs[i] = sums[i]/counts[i];
            }
        }
        isotonicRegression = new IsotonicRegression(locations, accs, counts);
    }

    public BucketInfo individualProbs(MultiLabelClfDataSet multiLabelClfDataSet){
        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    System.out.println(i);
                    double[] probs = imlGradientBoosting.predictClassProbs(multiLabelClfDataSet.getRow(i));
                    double[] calibratedProbs = IntStream.range(0, probs.length)
                            .mapToDouble(j->isotonicRegression.predict(probs[j])).toArray();
                    double[] count = new double[numBuckets];
                    double[] sum = new double[numBuckets];
                    double[] sumProbs = new double[numBuckets];
                    for (int a=0;a<probs.length;a++){
                        int index = (int)Math.floor(calibratedProbs[a]/bucketLength);
                        if (index<0){
                            index=0;
                        }
                        if (index>=numBuckets){
                            index = numBuckets-1;
                        }
                        count[index] += 1;
                        sumProbs[index] += calibratedProbs[a];
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(a)){
                            sum[index] += 1;
                        } else {
                            sum[index] += 0;
                        }
                    }
                    return new BucketInfo(count, sum, sumProbs);
                }).reduce(empty, BucketInfo::add, BucketInfo::add);
      return total;
    }



    public static class BucketInfo{

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

        static BucketInfo add(BucketInfo bucketInfo1, BucketInfo bucketInfo2){
            BucketInfo bucketInfo = new BucketInfo(bucketInfo1.counts.length);
            for (int i=0;i<bucketInfo1.counts.length;i++){
                bucketInfo.counts[i] = bucketInfo1.counts[i]+bucketInfo2.counts[i];
                bucketInfo.sums[i] = bucketInfo1.sums[i]+bucketInfo2.sums[i];
                bucketInfo.sumProbs[i] = bucketInfo1.sumProbs[i] + bucketInfo2.sumProbs[i];
            }
            return bucketInfo;
        }
    }








}
