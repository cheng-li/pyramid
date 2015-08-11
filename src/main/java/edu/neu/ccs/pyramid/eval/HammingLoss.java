package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/11/15.
 */
public class HammingLoss {


    /**
     * Hamming loss:
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @param multiLabels ground truth
     * @param predictions prediction
     * @return
     */
    public static double hammingLoss(MultiLabel[] multiLabels, MultiLabel[] predictions, int numLabels){

        int sysmetricDiffCount = 0;
        for (int i=0; i<multiLabels.length; i++) {
            MultiLabel label = multiLabels[i];
            MultiLabel prediction = predictions[i];

            sysmetricDiffCount += MultiLabel.symmetricDifference(label, prediction).size();
        }

        return sysmetricDiffCount * 1.0 / multiLabels.length / numLabels;
    }
}
