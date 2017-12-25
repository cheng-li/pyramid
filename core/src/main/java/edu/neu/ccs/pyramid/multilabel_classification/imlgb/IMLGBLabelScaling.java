package edu.neu.ccs.pyramid.multilabel_classification.imlgb;


import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.List;
import java.util.stream.IntStream;

public class IMLGBLabelScaling implements Serializable {
    private static final long serialVersionUID = 1L;
    private IMLGBIsotonicScaling imlgbIsotonicScaling;
    private IMLGradientBoosting imlGradientBoosting;

    public IMLGBLabelScaling(IMLGBIsotonicScaling imlgbIsotonicScaling) {
        this.imlgbIsotonicScaling = imlgbIsotonicScaling;
        this.imlGradientBoosting = imlgbIsotonicScaling.getBoosting();
    }

    private double[] calibratedProbs(Vector vector){
        double[] result = new double[imlGradientBoosting.getNumClasses()];
        double [] setProbs = imlGradientBoosting.predictAllAssignmentProbsWithConstraint(vector);
        double[] calibratedSetProbs = IntStream.range(0, setProbs.length)
                .mapToDouble(i->imlgbIsotonicScaling.calibratedProb(setProbs[i])).toArray();

        List<MultiLabel> allAssignments = imlGradientBoosting.getAssignments();
        for (int i=0;i<allAssignments.size();i++){
            MultiLabel multiLabel = allAssignments.get(i);
            for (int label:multiLabel.getMatchedLabels()){
                result[label] += calibratedSetProbs[i];
            }
        }
        return result;

    }

    public BucketInfo individualAccuracy(MultiLabelClfDataSet multiLabelClfDataSet){

        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;


       BucketInfo empty = new BucketInfo(numBuckets);
       BucketInfo total;
       total = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    System.out.println(i);
                    double[] probs = calibratedProbs(multiLabelClfDataSet.getRow(i));
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



