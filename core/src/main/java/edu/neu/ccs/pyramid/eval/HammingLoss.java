package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.stream.IntStream;

/**
 * Created by Rainicy on 8/11/15.
 */
public class HammingLoss {

    public static double hammingLoss(double tp, double tn, double fp, double fn){
        return (fp+fn)/(tp+tn+fp+fn);
    }

    public static double hammingLoss(MultiLabel label, MultiLabel prediction, int numLabels){
        return MultiLabel.symmetricDifference(label, prediction).size()/(double) numLabels;
    }

    /**
     * not divided by numLabels
     * @param label
     * @param prediction
     * @return
     */
    public static double unnormalized(MultiLabel label, MultiLabel prediction){
        return MultiLabel.symmetricDifference(label, prediction).size();
    }


    /**
     * Hamming loss:
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels ground truth
     * @param predictions prediction
     * @return
     */
    public static double hammingLoss(MultiLabel[] multiLabels, MultiLabel[] predictions, int numLabels){
        return IntStream.range(0,multiLabels.length).parallel()
                .mapToDouble(i -> hammingLoss(multiLabels[i], predictions[i], numLabels))
                .average().getAsDouble();
    }

    public static double hammingLoss(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        return hammingLoss(dataSet.getMultiLabels(),classifier.predict(dataSet),classifier.getNumClasses());
    }

}
