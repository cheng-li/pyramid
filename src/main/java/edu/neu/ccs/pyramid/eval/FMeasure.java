package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.List;

/**
 * follow the definition in http://en.wikipedia.org/wiki/Precision_and_recall
 * Created by chengli on 10/2/14.
 */
public class FMeasure {

    /**
     *
     * @param precision
     * @param recall
     * @return
     */
    public static double f1(double precision, double recall){
        return fBeta(precision,recall,1);
    }

    /**
     *
     * @param precision
     * @param recall
     * @param beta
     * @return
     */
    public static double fBeta(double precision, double recall, double beta){
        if (precision==0 || recall==0){
            return 0;
        }
        return (1+beta*beta)*precision*recall/(beta*beta*precision + recall);
    }

    /**
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double f1(MultiLabel[] multiLabels, List<MultiLabel> predictions) {

        double f = 0.0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions.get(i);
            f += MultiLabel.intersection(label, prediction).size() * 1.0 /
                    (label.getMatchedLabels().size() + prediction.getMatchedLabels().size());
        }

        return f * 2.0 / multiLabels.length;
    }
}
