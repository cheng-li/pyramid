package edu.neu.ccs.pyramid.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.eval.Recall;

/**
 * Created by chengli on 5/12/16.
 */
public class BinaryFMeasureTuner {

    public static double tuneThreshold(double[] probabilities, int[] labels, double beta){

        double bestThreshold = 0;
        double bestFBeta = Double.NEGATIVE_INFINITY;
        for (int a=0;a<=100;a++){
            double threshold = a*0.01;
            int[] pred = predictionByThreshold(probabilities, threshold);
            double precision = Precision.precision(labels,pred,1);
            double recall = Recall.recall(labels,pred,1);
            double fBeta = FMeasure.fBeta(precision,recall,beta);
            if (fBeta>bestFBeta){
                bestThreshold = threshold;
                bestFBeta = fBeta;
            }
        }
        return bestThreshold;
    }


    private static int[] predictionByThreshold(double[] probabilities, double threshold){
        int[] pred = new int[probabilities.length];
        for (int i=0;i<probabilities.length;i++){
            if (probabilities[i]>threshold){
                pred[i] = 1;
            }
        }
        return pred;
    }
}
