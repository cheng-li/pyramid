package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.ArgSort;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 8/21/15.
 */
public class AveragePrecision {


    /**
     * follows http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
     *
     * @param relevance the actual relevance of each item in the sorted list; sorted by the estimated relevance,
     *                  from most relevant to least relevant
     *
     * @return
     */
    public static double averagePrecision(boolean[] relevance){
        double totalRelevant = 0;
        double relevantSoFar = 0;
        double sumPrecisionAtK = 0;
        for (int i=0;i<relevance.length;i++){
            if (relevance[i]){
                totalRelevant += 1;
                relevantSoFar += 1;
                sumPrecisionAtK += relevantSoFar/(i+1);
            }
        }
        return SafeDivide.divide(sumPrecisionAtK,totalRelevant, 1);
    }


    /**
     * follows http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
     * https://en.wikipedia.org/wiki/Information_retrieval#Average_precision
     *
     * @param relevance the actual relevance of each item in the sorted list; sorted by the estimated relevance,
     *                  from most relevant to least relevant
     *                  the list is truncated so that some relevant items are not retrieved
     * @param totalNumRelevant total number of relevant items
     *
     * @return
     */
    public static double averagePrecisionTruncated(boolean[] relevance, int totalNumRelevant){
        double relevantSoFar = 0;
        double sumPrecisionAtK = 0;
        for (int i=0;i<relevance.length;i++){
            if (relevance[i]){
                relevantSoFar += 1;
                sumPrecisionAtK += relevantSoFar/(i+1);
            }
        }
        return SafeDivide.divide(sumPrecisionAtK,totalNumRelevant, 1);
    }

    public static double averagePrecision(int[] relevance){
        double totalRelevant = 0;
        double relevantSoFar = 0;
        double sumPrecisionAtK = 0;
        for (int i=0;i<relevance.length;i++){
            if (relevance[i]==1){
                totalRelevant += 1;
                relevantSoFar += 1;
                sumPrecisionAtK += relevantSoFar/(i+1);
            }
        }
        return SafeDivide.divide(sumPrecisionAtK,totalRelevant, 1);
    }

    public static double averagePrecisionTruncated(int[] relevance, int totalNumRelevant){
        double relevantSoFar = 0;
        double sumPrecisionAtK = 0;
        for (int i=0;i<relevance.length;i++){
            if (relevance[i]==1){
                relevantSoFar += 1;
                sumPrecisionAtK += relevantSoFar/(i+1);
            }
        }
        return SafeDivide.divide(sumPrecisionAtK,totalNumRelevant, 1);
    }

    /**
     * sort by scores, and compare against binary labels; bigger score is interpreted as more relevant
     * @param binaryLabels
     * @param scores
     * @return
     */
    public static double averagePrecision(int[] binaryLabels, double[] scores){
        int[] sortedIndices = ArgSort.argSortDescending(scores);
        int[] relevance = new int[binaryLabels.length];
        for (int i=0;i<relevance.length;i++){
            relevance[i] = binaryLabels[sortedIndices[i]];
        }
        return averagePrecision(relevance);
    }

    /**
     * sort by scores, and compare against binary labels; bigger score is interpreted as more relevant
     * @param binaryLabels
     * @param scores
     * @return
     */
    public static double averagePrecisionTruncated(int[] binaryLabels, double[] scores, int totalNumRelevant){
        int[] sortedIndices = ArgSort.argSortDescending(scores);
        int[] relevance = new int[binaryLabels.length];
        for (int i=0;i<relevance.length;i++){
            relevance[i] = binaryLabels[sortedIndices[i]];
        }
        return averagePrecisionTruncated(relevance, totalNumRelevant);
    }

    /**
     * compute average precision for a binary classification task
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double averagePrecision(Classifier.ProbabilityEstimator classifier, ClfDataSet dataSet){
        if (classifier.getNumClasses()!=2){
            throw new IllegalArgumentException("classifier.getNumClasses()!=2");
        }
        return averagePrecision(classifier, dataSet, dataSet.getLabels());
    }

    public static double averagePrecision(Classifier.ProbabilityEstimator classifier, DataSet dataSet, int[] labels){
        double[] probs = new double[dataSet.getNumDataPoints()];
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->probs[i]= classifier.predictClassProbs(dataSet.getRow(i))[1]);
        return averagePrecision(labels, probs);
    }


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
