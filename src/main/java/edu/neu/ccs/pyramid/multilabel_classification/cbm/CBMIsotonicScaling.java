package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.IsotonicRegression;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 1/21/18
 *
 * Follows IMLGIsotonicScaling calss in package:
 * package edu.neu.ccs.pyramid.multilabel_classification.imlgb;
 */
public class CBMIsotonicScaling {

    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private CBM cbm;

    public CBMIsotonicScaling(CBM cbm, MultiLabelClfDataSet dataSet) {
        this.cbm = cbm;

        List<MultiLabel> allAssignments = cbm.getSupport();

        final int numBuckets = 10000;
        double bucketLength = 1.0/numBuckets;
        double[] locations = new double[numBuckets];
        for (int i=0; i<numBuckets; i++) {
            locations[i] = i*bucketLength + 0.5 * bucketLength;
        }

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> {
                    double[] probs = cbm.predictAssignmentProbs(dataSet.getRow(i),
                            allAssignments);
                    double[] count = new double[numBuckets];
                    double[] sum = new double[numBuckets];
                    for (int a=0; a<probs.length; a++) {
                        int index = (int) Math.floor(probs[a]/bucketLength);
                        if (index < 0) index = 0;
                        if (index >= numBuckets) index = numBuckets-1;
                        count[index] += 1;
                        if (allAssignments.get(a).equals(dataSet.getMultiLabels()[i])) {
                            sum[index] += 1;
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

    public CBM getCBM() {
        return cbm;
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel) {
        double uncalibrated = cbm.predictAssignmentProb(vector, multiLabel);
        return isotonicRegression.predict(uncalibrated);
    }

    public double calibratedProb(double uncalibratedProb) {
        return isotonicRegression.predict(uncalibratedProb);
    }

    private static class BucketInfo{

        private BucketInfo(int size) {
            counts = new double[size];
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
