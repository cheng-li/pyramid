package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.List;

/**
 * Created by chengli on 10/2/14.
 */
public class Recall {

    public static double recall(double tp, double fn) {
        return SafeDivide.divide(tp,tp+fn,1);
    }


    /**
     *
     * @param classifier
     * @param dataSet
     * @param k class index
     * @return
     */
    public static double recall(Classifier classifier, ClfDataSet dataSet, int k){
        int[] labels = dataSet.getLabels();
        int[] predictions = classifier.predict(dataSet);
        return recall(labels,predictions,k);
    }

    /**
     *
     * @param labels
     * @param predictions
     * @param k class index
     * @return
     */
    public static double recall(int[] labels, int[] predictions, int k){
        int falseNegative = 0;
        int truePositives = 0;
        for (int i=0;i<labels.length;i++){
            if (labels[i]==k){
                if (predictions[i]==k){
                    truePositives += 1;
                } else {
                    falseNegative += 1;
                }
            }
        }
        return recall(truePositives,falseNegative);
    }

    @Deprecated
    /**
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double recall(MultiLabel[] multiLabels, MultiLabel[] predictions) {
        double r = 0.0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions[i];
            if (label.getMatchedLabels().size() == 0){
                r += 1.0;
            } else {
                r += MultiLabel.intersection(label, prediction).size() * 1.0 / label.getMatchedLabels().size();
            }
        }
        return r / multiLabels.length;
    }

    public static double recall(MultiLabel label, MultiLabel prediction) {

        if (label.getMatchedLabels().size() == 0){
            return 1.0;
        } else {
            return MultiLabel.intersection(label, prediction).size() * 1.0 / label.getMatchedLabels().size();
        }
    }






    @Deprecated
    /**
     * Please see function: double recall(MultiLabel[] multiLabels, List<MultiLabel> predictions);
     * @param classifier
     * @param dataset
     * @return
     */
    public static double recall(MultiLabelClassifier classifier, MultiLabelClfDataSet dataset) {
        return recall(dataset.getMultiLabels(), classifier.predict(dataset));
    }

}
