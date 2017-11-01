package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.List;
import java.util.stream.IntStream;

public class IMLGBIsotonicScaling implements Serializable{
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private IMLGradientBoosting boosting;

    public IMLGBIsotonicScaling(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
//        System.out.println("calibrating with isotonic regression");
        this.boosting = boosting;
        List<MultiLabel> allAssignments = boosting.getAssignments();


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
                    double[] probs = boosting.predictAllAssignmentProbsWithConstraint(multiLabelClfDataSet.getRow(i));
                    double[] count = new double[numBuckets];
                    double[] sum = new double[numBuckets];
                    for (int a=0;a<probs.length;a++){
                        int index = (int)Math.floor(probs[a]/bucketLength);
                        if (index<0){
                            index=0;
                        }
                        if (index>=numBuckets){
                            index = numBuckets-1;
                        }
                        count[index] += 1;
                        if (allAssignments.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])){
                            sum[index] += 1;
                        } else {
                            sum[index] += 0;
                        }
                    }
                    return new BucketInfo(count, sum);
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
//        System.out.println("calibration done");
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double uncalibrated = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
        return isotonicRegression.predict(uncalibrated);
    }

    public double calibratedProb(double uncalibratedProb){
        return isotonicRegression.predict(uncalibratedProb);
    }

    private static class BucketInfo{

        public BucketInfo(int size) {
            counts = new double[size];
            sums = new double[size];
        }

        public BucketInfo(double[] counts, double[] sums) {
            this.counts = counts;
            this.sums = sums;
        }

        double[] counts;
        double[] sums;

        static BucketInfo add(BucketInfo bucketInfo1, BucketInfo bucketInfo2){
            BucketInfo bucketInfo = new BucketInfo(bucketInfo1.counts.length);
            for (int i=0;i<bucketInfo1.counts.length;i++){
                bucketInfo.counts[i] = bucketInfo1.counts[i]+bucketInfo2.counts[i];
                bucketInfo.sums[i] = bucketInfo1.sums[i]+bucketInfo2.sums[i];
            }
            return bucketInfo;
        }
    }
}
