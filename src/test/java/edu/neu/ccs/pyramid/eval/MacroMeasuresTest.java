package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/12/15.
 */
public class MacroMeasuresTest {
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

        MacroMeasures macroMeasures = new MacroMeasures(3);
        macroMeasures.update(labels,predictions);

        System.out.println("Expected Macro-Precision: 0.22222222222222221");
        System.out.println("Expected Macro-Recall: 0.33333333333333331");
        System.out.println("Expected Macro-F1: 0.26666666666666666");
        System.out.println(macroMeasures);
        System.out.println("Expected Macro-F-Beta=0.5: 0.23809523809523805");
        System.out.println("Macro-FBeta=0.5: " + macroMeasures.getFScore(0.5));
    }
}
