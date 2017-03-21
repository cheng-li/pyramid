package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.ArgSort;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/2/14.
 */
public class Precision {

    public static double precision(double tp, double fp) {
        return SafeDivide.divide(tp,tp+fp,1);
    }

    /**
     *
     * @param classifier
     * @param dataSet
     * @param k class index
     * @return
     */
    public static double precision(Classifier classifier, ClfDataSet dataSet, int k){
        int[] labels = dataSet.getLabels();
        int[] predictions = classifier.predict(dataSet);
        return precision(labels,predictions,k);
    }

    /**
     *
     * @param labels
     * @param predictions
     * @param k class index
     * @return
     */
    public static double precision(int[] labels, int[] predictions, int k){
        int falsePositive = 0;
        int truePositive = 0;
        for (int i=0;i<labels.length;i++){
            if (predictions[i]==k){

                if (labels[i]==k){
                    truePositive += 1;
                } else {
                    falsePositive += 1;
                }
            }
        }
        return precision(truePositive,falsePositive);
    }

    /**
     * https://manikvarma.github.io/downloads/XC/XMLRepository.html#Prabhu14
     * @param scores
     * @param groudtruth
     * @param k
     * @return
     */
    public static double precisionAtK(double[] scores, MultiLabel groudtruth, int k){
        double total = 0;
        int[] top = ArgSort.argSortDescending(scores);
        for (int r=0;r<k;r++){
            int l = top[r];
            if (groudtruth.matchClass(l)){
                total += 1;
            }
        }
        return total/k;
    }

    public static double precisionAtK(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, int k){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i-> precisionAtK(multiLabelClassifier.predictClassProbs(dataSet.getRow(i)),dataSet.getMultiLabels()[i],k))
                .average().getAsDouble();
    }

    @Deprecated
    /**
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double precision(MultiLabel[] multiLabels, MultiLabel[] predictions){

        double p = 0.0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions[i];
            if (prediction.getMatchedLabels().size() == 0){
                p += 1.0;
            } else {
                p += MultiLabel.intersection(label, prediction).size() * 1.0 / prediction.getMatchedLabels().size();
            }
        }

        return p / multiLabels.length;
    }

    @Deprecated
    /**
     * see function: double precision(MultiLabel[] multiLabels, List<MultiLabel> predictions)
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double precision(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        return precision(dataSet.getMultiLabels(),classifier.predict(dataSet));
    }
}
