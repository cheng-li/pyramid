package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;

import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class Accuracy {

    public static double accuracy(int tp, int tn, int fp, int fn){
        return (tp+tn)*1.0/(tp+tn+fp+fn);
    }

    /**
     * multi-threaded
     */
    public static double accuracy(Classifier classifier, ClfDataSet clfDataSet){
        int[] prediction = classifier.predict(clfDataSet);
        return accuracy(clfDataSet.getLabels(),prediction);
    }

    /**
     * multi-threaded
     */
    public static double accuracy(int[] labels, int[] predictions){
        if (labels.length!=predictions.length){
            throw new IllegalArgumentException("labels.length!=predictions.length");
        }
        double numCorrect =  IntStream.range(0,labels.length).parallel()
                .filter(i-> labels[i]==predictions[i])
                .count();
        return numCorrect/labels.length;
    }


    /**
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * Exact match ratio.
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double accuracy(MultiLabel[] multiLabels, MultiLabel[] predictions){
        if (multiLabels.length == 0) {
            throw new IllegalArgumentException("multi labels length is zero.");
        }
        if (multiLabels.length != predictions.length) {
            throw new IllegalArgumentException("multi labels length is not equal to predictions length.");
        }

        double numCorrect = IntStream.range(0,multiLabels.length).parallel()
                .filter(i -> multiLabels[i].equals(predictions[i]))
                .count();
        return numCorrect/multiLabels.length;
    }

    public static double accuracy(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        return accuracy(dataSet.getMultiLabels(), classifier.predict(dataSet));
    }

    /**
     * proportion of the predicted correct labels to the total number of labels for that instance.
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double partialAccuracy(MultiLabel[] multiLabels, MultiLabel[] predictions) {
        double a = 0.0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions[i];
            a += MultiLabel.intersection(label, prediction).size() * 1.0 / MultiLabel.union(label, prediction).size();
        }
        return a / multiLabels.length;
    }

    public static double partialAccuracy(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet) {
        return partialAccuracy(dataSet.getMultiLabels(), classifier.predict(dataSet));
    }
}
