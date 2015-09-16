package edu.neu.ccs.pyramid.eval;

/**
 * Created by Rainicy on 8/11/15.
 */
public class ConfusionMatrixMeasures {

    public static double accuracy(int tp, int tn, int fp, int fn){
        return (tp+tn)*1.0/(tp+tn+fp+fn);
    }

    /**
     * Returns the precision.
     * @param tp true positives
     * @param fp false positives
     * @return precision
     */
    public static double precision(int tp, int fp) {

        if (tp + fp == 0) {
            return 1;
        }

        return tp * 1.0 / (tp + fp);
    }


    /**
     * Returns the recall.
     * @param tp true positives
     * @param fn false negatives
     * @return
     */
    public static double recall(int tp, int fn) {

        if (tp + fn == 0) {
            return 1;
        }

        return tp * 1.0 / (tp + fn);
    }

    /**
     * Returns the specificity.
     * @param tn true negaives
     * @param fp false positives
     * @return the specificity.
     */
    public static double specificity(int tn, int fp) {

        if (tn + fp == 0) {
            return 1;
        }

        return tn * 1.0 / (tn + fp);
    }

    public static double fScore(int tp, int fp, int fn, double beta) {
        double precision = precision(tp, fp);
        double recall = recall(tp, fn);
        return FMeasure.fBeta(precision,recall,beta);

    }



}
