package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;

/**
 * Created by chengli on 10/2/14.
 */
public class Recall {

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
        double positives = 0;
        double truePositives = 0;
        for (int i=0;i<labels.length;i++){
            if (labels[i]==k){
                positives += 1;
                if (predictions[i]==k){
                    truePositives += 1;
                }
            }
        }
        return truePositives/positives;
    }
}
