package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;

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
}
