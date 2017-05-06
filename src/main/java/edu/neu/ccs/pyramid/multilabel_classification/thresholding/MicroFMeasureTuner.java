package edu.neu.ccs.pyramid.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.MLConfusionMatrix;
import edu.neu.ccs.pyramid.eval.MicroAverage;

import java.util.stream.IntStream;

/**
 * Created by chengli on 5/6/17.
 */
public class MicroFMeasureTuner {

    public static double tuneThreshold(double[][] probabilities, MultiLabel[] groundTruth){
        double bestThreshold = 0;
        double bestFBeta = Double.NEGATIVE_INFINITY;
        for (int a=0;a<=100;a++) {
            double threshold = a * 0.01;
            double f = evalThreshold(probabilities, groundTruth, threshold);
            if (f>bestFBeta){
                bestThreshold = threshold;
                bestFBeta = f;
            }
        }
        return bestThreshold;
    }




    private static double evalThreshold(double[][] probabilities, MultiLabel[] groundTruth, double threshold){
        int numData = probabilities.length;
        int numClasses = probabilities[0].length;
        MultiLabel[] prediction = new MultiLabel[probabilities.length];
        IntStream.range(0, numData).parallel().forEach(i->{
            prediction[i] = new MultiLabel();
            for (int l=0;l<numClasses;l++){
                if (probabilities[i][l]>threshold){
                    prediction[i].addLabel(l);
                }
            }
        });

        MLConfusionMatrix mlConfusionMatrix = new MLConfusionMatrix(numClasses, groundTruth, prediction);
        return new MicroAverage(mlConfusionMatrix).getF1();
    }


}
