package edu.neu.ccs.pyramid.core.eval;


import java.util.stream.IntStream;

/**
 * Created by Rainicy on 8/11/15.
 */
public class MicroMeasures extends LabelBasedMeasures {
    /**
     * Construct function: initialize each variables.
     *
     * @param numLabels
     */
    public MicroMeasures(int numLabels) {
        super(numLabels);
    }

    /**
     * Returns the average Micro precision for all labels.
     * @return average precision
     */
    public double getPrecision() {
        int tp = IntStream.of(truePositives).sum();
        int fp = IntStream.of(falsePositives).sum();
        return Precision.precision(tp,fp);
    }

    /**
     * Returns the average Micro recall for all labels.
     * @return average recall
     */
    public double getRecall() {
        int tp = IntStream.of(truePositives).sum();
        int fn = IntStream.of(falseNegatives).sum();
        return Recall.recall(tp,fn);
    }


    /**
     * Returns the average Micro specificity for all labels.
     * @return average specificity
     */
    public double getSpecificity() {
        int tn = IntStream.of(trueNegatives).sum();
        int fp = IntStream.of(falsePositives).sum();

        return ConfusionMatrixMeasures.specificity(tn,fp);
    }


    /**
     * Returns the average Micro F score for all labels.
     * @param beta
     * @return average F score
     */
    public double getFScore(double beta) {
        int tp = IntStream.of(truePositives).sum();
        int fp = IntStream.of(falsePositives).sum();
        int fn = IntStream.of(falseNegatives).sum();
        return FMeasure.fBeta(tp,fp,fn,beta);
    }

    /**
     * Returns the average Micro F1 score for all labels.
     * @return average F1 score
     */
    public double getF1Score() {
        return getFScore(1.0);
    }

    @Override
    public String toString() {
        return "Micro-Precision: \t" + getPrecision() +
                "\nMicro-Recall: \t" + getRecall() +
                "\nMicro-Specificity: \t" + getSpecificity() +
                "\nMicro-F1Score: \t" + getF1Score();
    }

}
