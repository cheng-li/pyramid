package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class IMLGBLabelIsotonicScaling implements Serializable {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    List<IsotonicRegression> isotonicRegressionList;

    public IMLGBLabelIsotonicScaling(IMLGradientBoosting imlGradientBoosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        this.imlGradientBoosting = imlGradientBoosting;
        this.isotonicRegressionList = new ArrayList<>();
        for (int l= 0; l < imlGradientBoosting.getNumClasses(); l++) {
            final int calssIndex = l;
            final int numBuckets = 10000;
            double bucketLength = 1.0 / numBuckets;
            double[] locations = new double[numBuckets];
            for (int j = 0; j < numBuckets; j++) {
                locations[j] = j * bucketLength + 0.5 * bucketLength;
            }
            BucketInfo empty = new BucketInfo(numBuckets);
            BucketInfo total;
            total = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .mapToObj(i -> {
                        double prob = imlGradientBoosting.predictClassProb(multiLabelClfDataSet.getRow(i), calssIndex);
                        double[] count = new double[numBuckets];
                        double[] sum = new double[numBuckets];
                        double[] sumProbs = new double[numBuckets];
                        int index = (int) Math.floor(prob / bucketLength);
                        if (index < 0) {
                            index = 0;
                        }
                        if (index >= numBuckets) {
                            index = numBuckets - 1;
                        }
                        count[index] += 1;
                        sumProbs[index] += prob;
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(calssIndex)) {
                            sum[index] += 1;
                        } else {
                            sum[index] += 0;
                        }

                        return new BucketInfo(count, sum, sumProbs);
                    }).reduce(empty, BucketInfo::add, BucketInfo::add);
            double[] counts = total.counts;
            double[] sums = total.sums;
            double[] accs = new double[counts.length];
            for (int k = 0; k < counts.length; k++) {
                if (counts[k] != 0) {
                    accs[k] = sums[k] / counts[k];
                }
            }
            IsotonicRegression isotonicRegression = new IsotonicRegression(locations, accs, counts);
            isotonicRegressionList.add(isotonicRegression);
        }
    }


    public double calibratedClassProb(double prob, int labelIndex){
        return isotonicRegressionList.get(labelIndex).predict(prob);
    }

    public double[] calibratedClassProbs(double[]probs){
        return IntStream.range(0, probs.length).mapToDouble(j->calibratedClassProb(probs[j], j)).toArray();

    }


    public BucketInfo getBucketInfo(MultiLabelClfDataSet multiLabelClfDataSet){
        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    double[] probs = imlGradientBoosting.predictClassProbs(multiLabelClfDataSet.getRow(i));
                    double[] calibratedProbs = IntStream.range(0, probs.length)
                            .mapToDouble(j->isotonicRegressionList.get(j).predict(probs[j])).toArray();
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



}
