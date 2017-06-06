package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;

/**
 * follow the definition in http://en.wikipedia.org/wiki/Precision_and_recall
 * Created by chengli on 10/2/14.
 */
public class FMeasure {

    public static double fBeta(double tp, double fp, double fn, double beta){
        double precision = Precision.precision(tp,fp);
        double recall = Recall.recall(tp,fn);
        return fBeta(precision,recall,beta);
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


    public static double f1(double tp, double fp, double fn){
        return fBeta(tp,fp,fn,1);
    }

    /**
     *
     * @param precision
     * @param recall
     * @return
     */
    public static double f1(double precision, double recall){
        return fBeta(precision,recall,1);
    }


    @Deprecated
    /**
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels
     * @param predictions
     * @return
     */
    public static double f1(MultiLabel[] multiLabels, MultiLabel[] predictions) {

        double f = 0.0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions[i];
            f += MultiLabel.intersection(label, prediction).size() * 2.0 /
                    (label.getMatchedLabels().size() + prediction.getMatchedLabels().size());
        }

        return f / multiLabels.length;
    }
}
