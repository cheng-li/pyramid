package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class Accuracy {

    /**
     * multi-threaded
     */
    public static double accuracy(Classifier classifier, ClfDataSet clfDataSet){
        int[] prediction = classifier.predict(clfDataSet);
        return accuracy(clfDataSet.getLabels(),prediction);
    }

    public static double accuracy(int[] labels, int[] predictions){
        double numCorrect = 0;
        if (labels.length!=predictions.length){
            throw new IllegalArgumentException("labels.length!=predictions.length");
        }
        for (int i=0;i<labels.length;i++){
            if (labels[i] == predictions[i]){
                numCorrect += 1;
            }
        }
        return numCorrect/labels.length;
    }


}
