package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/12/15.
 */
public class MicroMeasuresTest {
    public static void main(String[] args) {
        MultiLabel[] labels = new MultiLabel[6];
        MultiLabel[] predictions = new MultiLabel[6];
        labels[0] = new MultiLabel();
        labels[1] = new MultiLabel();
        labels[2] = new MultiLabel();
        labels[3] = new MultiLabel();
        labels[4] = new MultiLabel();
        labels[5] = new MultiLabel();

        predictions[0] = new MultiLabel();
        predictions[1] = new MultiLabel();
        predictions[2] = new MultiLabel();
        predictions[3] = new MultiLabel();
        predictions[4] = new MultiLabel();
        predictions[5] = new MultiLabel();

        labels[0].addLabel(0);
        labels[1].addLabel(1);
        labels[2].addLabel(2);
        labels[3].addLabel(0);
        labels[4].addLabel(1);
        labels[5].addLabel(2);

        predictions[0].addLabel(0);
        predictions[1].addLabel(2);
        predictions[2].addLabel(1);
        predictions[3].addLabel(0);
        predictions[4].addLabel(0);
        predictions[5].addLabel(1);

        MicroMeasures microMeasures = new MicroMeasures(3);
        microMeasures.update(labels,predictions);

        System.out.println("Expected Micro-Precision: 0.33333333333333331");
        System.out.println("Expected Micro-Recall: 0.33333333333333331");
        System.out.println("Expected Micro-F1: .33333333333333331");
        System.out.println(microMeasures);
        System.out.println("Expected Macro-F-Beta=0.5: 0.33333333333333331");
        System.out.println("Micro-FBeta=0.5: " + microMeasures.getFScore(0.5));
    }
}
