package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.util.ArgSort;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by Rainicy on 8/21/15.
 */
public class AveragePrecision {


    /**
     * Average Precision based on each data sample. First, calculate the average
     * precision for each data, then take average.
     * For calculating the average precision of each data point, please refer:
     * http://www.ccs.neu.edu/course/cs6200sp15/slides/m06.s03%20-%20batch%20evaluation%20measures.pdf
     *
     * @param classifier
     * @param dataSet
     * @return average precision cross all data samples.
     */
    public static double averagePrecision(MultiLabelClassifier.ClassScoreEstimator classifier, MultiLabelClfDataSet dataSet){

        double ap = 0.0;

        MultiLabel[] labels = dataSet.getMultiLabels();

        for (int i=0; i<labels.length; i++) {
            Set<Integer> label = labels[i].getMatchedLabels();
            double[] scores = classifier.predictClassScores(dataSet.getRow(i));
            ap += averagePrecision(label, scores);
        }
        return ap * 1.0 / labels.length;
    }

    /**
     * Average Precision for each data point, based by given scores.
     * @param label
     * @param scores
     * @return average precision per data point.
     */
    private static double averagePrecision(Set<Integer> label, double[] scores) {

        // sorted the labels by scores.(Descending)
        int[] sortedIndices = ArgSort.argSortDescending(scores);

        double sumPrecision = 0.0;
        // the precision at k, k is the cutoff from top scores to bottom.
//        double[] precisionAtK = new double[scores.length];

        Set<Integer> positivePredict = new HashSet<>();
        for (int k=0; k<sortedIndices.length; k++) {
            int predict = sortedIndices[k];
            positivePredict.add(predict);  // add the current label into prediction positive set.
            if (label.contains(predict)) {
                sumPrecision += getPrecision(label, positivePredict);
            }
        }

        return 1.0 / label.size() * sumPrecision;
    }

    private static double getPrecision(Set<Integer> label, Set<Integer> positivePredict) {
        int tp = 0;
        int fp = 0;
        int fn = 0;

        for (Integer predict : positivePredict) {
            if (label.contains(predict)) {
                tp++;
            }
        }
        fp = positivePredict.size() - tp;
        fn = label.size() - tp;

        return Precision.precision(tp,fp);
    }


}
