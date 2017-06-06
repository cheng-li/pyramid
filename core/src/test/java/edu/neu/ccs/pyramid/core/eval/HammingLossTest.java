package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/12/15.
 */
public class HammingLossTest {
    /**
     * Test example comes from:
     * http://scikit-learn.org/stable/modules/model_evaluation.html#hamming-loss
     * @param args
     */
    public static void main(String[] args) {

        MultiLabel[] labels = new MultiLabel[1];
        MultiLabel[] predictions = new MultiLabel[1];

        labels[0] = new MultiLabel();
        predictions[0] = new MultiLabel();
        labels[0].addLabel(1);
        labels[0].addLabel(2);
        labels[0].addLabel(3);
        labels[0].addLabel(4);
        predictions[0].addLabel(2);
        predictions[0].addLabel(2);
        predictions[0].addLabel(3);
        predictions[0].addLabel(4);

        System.out.println("Expected (value=0.25) - Output: " + HammingLoss.hammingLoss(labels, predictions, 4));

        MultiLabel[] labels1 = new MultiLabel[2];
        MultiLabel[] predictions1 = new MultiLabel[2];
        labels1[0] = new MultiLabel();
        labels1[1] = new MultiLabel();
        predictions1[0] = new MultiLabel();
        predictions1[1] = new MultiLabel();

        labels1[0].addLabel(0);
        labels1[0].addLabel(1);
        labels1[1].addLabel(1);

        predictions1[0].addLabel(0);
        predictions1[1].addLabel(0);

        System.out.println("Expected (value=0.75) - Output: " + HammingLoss.hammingLoss(labels1, predictions1, 2));


    }
}
