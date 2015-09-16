package edu.neu.ccs.pyramid.eval;

/**
 * Created by Rainicy on 8/11/15.
 */
public class ConfusionMatrixMeasures {





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





}
