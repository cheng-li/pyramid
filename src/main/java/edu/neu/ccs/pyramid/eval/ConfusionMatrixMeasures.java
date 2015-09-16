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
     * @param fn false negatives
     * @return precision
     */
    public static double precision(int tp, int fp, int fn) {

        /* if the size of ground truth is 0, and
         * the size of test outcome is also 0,
         * we consider the precision is 1.
         */
        if (tp + fp + fn == 0) {
            return 1;
        }

        /*
         * if the size of ground truth is not 0, but
         * the size of test outcome is 0,
         * we consider the precision is 0.
         */
        if (tp + fp == 0) {
            return 0;
        }

        return tp * 1.0 / (tp + fp);
    }


    /**
     * Returns the recall.
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @return
     */
    public static double recall(int tp, int fp, int fn) {
        /* if the size of ground truth is 0, and
         * the size of test outcome is also 0,
         * we consider the recall is 1.
         */
        if (tp + fp + fn == 0) {
            return 1;
        }

        /*
         * if the size of ground truth is 0, but
         * the size of test outcome is not 0,
         * we consider the precision is 0.
         */
        if (tp + fn == 0) {
            return 0;
        }

        return tp * 1.0 / (tp + fn);
    }

    /**
     * Returns the specificity.
     * @param tn true negaives
     * @param fp false positives
     * @param fn false negatives
     * @return the specificity.
     */
    public static double specificity(int tn, int fp, int fn) {

        /*
         * only has tp. then consider specificity as 1.
         */
        if (tn + fp + fn == 0) {
            return 1;
        }

        /*
         * if tn + fp is 0, then consider specificity as 0.
         */
        if (tn + fp == 0) {
            return 0;
        }

        return tn * 1.0 / (tn + fp);
    }

    /**
     * return the fScore.
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @param beta ratio of recall compared to precision.
     * @return
     */
    public static double fScore(int tp, int fp, int fn, double beta) {
        /*
         * if the size of ground truth is 0 and the size of
         * test outcomes is 0, then consider the F score is 1.
         */
        if (tp + fp + fn == 0) {
            return 1;
        }

        double betaSqr = beta * beta;

        return (1.0 + betaSqr) * tp / ( (1.0 + betaSqr) * tp + betaSqr * fn + fp );
    }

    /**
     * return the F1 Score.
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @return F1 score.
     */
    public static double f1Score(int tp, int fp, int fn) {
        return fScore(tp, fp, fn, 1.0);
    }

}
