package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.IsotonicRegression;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 1/31/18
 */
public class PMatrixIsotonicScaling {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private CMLCRF cmlcrf;

    public PMatrixIsotonicScaling(CMLCRF cmlcrf, MultiLabelClfDataSet dataSet) {
        this.cmlcrf = cmlcrf;

        List<MultiLabel> allAssignments = cmlcrf.getSupportCombinations();

        final int numBuckets = 10000;
        double bucketLength = 1.0/numBuckets;
        double[] locations = new double[numBuckets];
        for (int i=0; i<numBuckets; i++) {
            // TODO: what's this?
            locations[i] = i*bucketLength + 0.5 * bucketLength;
        }


        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> {
                    double[] probs = cmlcrf.predictAssignmentProbs(dataSet.getRow(i), allAssignments);
                    GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
                    double[][] p = generalF1Predictor.getPMatrix(cmlcrf.getNumClasses(), allAssignments,
                            DoubleStream.of(probs).boxed().collect(Collectors.toList()));

                    double[] count = new double[numBuckets];
                    double[] sum = new double[numBuckets];
                    for (int l=0; l<p.length; l++) {
                        for (int s=0; s<p[l].length; s++) {
                            int index = (int) Math.floor(p[l][s]/bucketLength);
                            if (index < 0) index = 0;
                            if (index >= numBuckets) index = numBuckets - 1;
                            count[index] += 1;
                            MultiLabel groundTruth = dataSet.getMultiLabels()[i];
                            // TODO: check the geNumMatchedLabels is the right thing here.
                            if (groundTruth.matchClass(l) && groundTruth.getNumMatchedLabels() == s+1) {
                                sum[index] += 1;
                            }
                        }
                    }

                    return new BucketInfo(count, sum);
                }).reduce(empty, BucketInfo::add, BucketInfo::add);
        double[] counts = total.counts;
        double[] sums = total.sums;
        double[] accs = new double[counts.length];
        for (int i=0; i<counts.length; i++) {
            if (counts[i] != 0) {
                accs[i] = sums[i]/counts[i];
            }
        }
        isotonicRegression = new IsotonicRegression(locations, accs, counts);
    }

    public CMLCRF getCmlcrf() {
        return cmlcrf;
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel) {
        double uncalibrated = cmlcrf.predictAssignmentProb(vector, multiLabel);
        return isotonicRegression.predict(uncalibrated);
    }

    public double calibratedProb(double uncalibratedProb) {
        return isotonicRegression.predict(uncalibratedProb);
    }


    private static class BucketInfo{

        private BucketInfo(int size) {
            // store the probs counts
            counts = new double[size];
            // store the correct counts
            sums = new double[size];
        }

        private BucketInfo(double[] counts, double[] sums) {
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
