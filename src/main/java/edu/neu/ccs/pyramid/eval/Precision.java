package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/2/14.
 */
public class Precision {

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
        double predictedPositive = 0;
        double truePositive = 0;
        for (int i=0;i<labels.length;i++){
            if (predictions[i]==k){
                predictedPositive += 1;
                if (labels[i]==k){
                    truePositive += 1;
                }
            }
        }
        return truePositive/predictedPositive;
    }

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
            p += MultiLabel.intersection(label, prediction).size() * 1.0 / prediction.getMatchedLabels().size();
        }

        return p / multiLabels.length;
    }

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
