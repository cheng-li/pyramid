package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;

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
