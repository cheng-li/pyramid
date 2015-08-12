package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/11/15.
 */
public abstract class LabelBasedMeasures {

    /**
     * number of unique labels.
     */
    protected int numLabels;

    /**
     * for each label(index of the array),
     * the number of true positives.
     */
    protected int[] truePositives;

    /**
     * for each label(index of the array),
     * the number of true negatives.
     */
    protected int[] trueNegatives;

    /**
     * for each label(index of the array),
     * the number of false positives
     */
    protected int[] falsePositives;

    /**
     * for each label(index of the array),
     * the number of false negatives.
     */
    protected int[] falseNegatives;


    /**
     * Construct function: initialize each variables.
     * @param numLabels
     */
    public LabelBasedMeasures(int numLabels) {
        if (numLabels == 0) {
            throw new RuntimeException("initialization with zero label.");
        }
        this.numLabels = numLabels;

        truePositives = new int[numLabels];
        falsePositives = new int[numLabels];
        trueNegatives = new int[numLabels];
        falseNegatives = new int[numLabels];
    }


    /**
     * update the confusion matrix by given one sample
     * ground truth and prediction.
     * @param label ground truth
     * @param prediction predictions
     */
    public void update(MultiLabel label, MultiLabel prediction) {

        for (int i=0; i<numLabels; i++) {
            boolean actual = label.matchClass(i);
            boolean predicted = prediction.matchClass(i);

            if (actual) {
                if (predicted) {
                    truePositives[i]++;
                } else {
                    falseNegatives[i]++;
                }
            } else {
                if (predicted) {
                    falsePositives[i]++;
                } else {
                    trueNegatives[i]++;
                }
            }
        }
    }

    /**
     * update the confusion matrix by given an array of ground truth and
     * predictions.
     * @param labels ground truth array
     * @param predictions prediction array
     */
    public void update(MultiLabel[] labels, MultiLabel[] predictions) {

        if (labels.length == 0) {
            throw new RuntimeException("Empty given ground truth.");
        }
        if (labels.length != predictions.length) {
            throw new RuntimeException("The lengths of ground truth and predictions should" +
                    "be the same.");
        }

        for (int i=0; i<labels.length; ++i) {
            update(labels[i], predictions[i]);
        }
    }

}
