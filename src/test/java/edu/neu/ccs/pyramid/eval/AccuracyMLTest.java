package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by Rainicy on 8/11/15.
 */
public class AccuracyMLTest {

    public static void main(String[] args) {
        MultiLabel[] labels = new MultiLabel[1];
        MultiLabel[] predictions = new MultiLabel[1];

        labels[0] = new MultiLabel();
        labels[0].addLabel(0);
        labels[0].addLabel(1);
        predictions[0] = new MultiLabel();
        predictions[0].addLabel(0);
        predictions[0].addLabel(1);
        System.out.println("Expected: (value=1.0) - Output: " + Accuracy.accuracy(labels, predictions));

        predictions[0].addLabel(2);
        System.out.println("Expected: (value=0.0) - Output:" + Accuracy.accuracy(labels, predictions));


        int labelLength = 1000;
        MultiLabel[] labels1 = new MultiLabel[labelLength];
        MultiLabel[] predictions1 = new MultiLabel[labelLength];
        for (int i=0; i<labelLength; i++) {
            labels1[i] = new MultiLabel();
            predictions1[i] = new MultiLabel();

            labels1[i].addLabel(1);
            labels1[i].addLabel(2);
            predictions1[i].addLabel(1);
            predictions1[i].addLabel(2);
        }
        System.out.println("Expected: (value=1.0) - Output: " + Accuracy.accuracy(labels1, predictions1));

        MultiLabel[] labels2 = new MultiLabel[labelLength];
        MultiLabel[] predictions2 = new MultiLabel[labelLength];
        for (int i=0; i<labelLength; i++) {
            labels2[i] = new MultiLabel();
            predictions2[i] = new MultiLabel();

            labels2[i].addLabel(0);
            predictions2[i].addLabel(0);
        }
        System.out.println("Expected: (value=1.0) - Output: " + Accuracy.accuracy(labels2, predictions2));

        MultiLabel[] labels3 = new MultiLabel[labelLength];
        MultiLabel[] predictions3 = new MultiLabel[labelLength];
        for (int i=0; i<labelLength; i++) {
            labels3[i] = new MultiLabel();
            predictions3[i] = new MultiLabel();
        }
        System.out.println("Expected: (value=1.0) - Output: " + Accuracy.accuracy(labels3, predictions3));


        MultiLabel[] labels4 = new MultiLabel[labelLength];
        MultiLabel[] predictions4 = new MultiLabel[labelLength];
        for (int i=0; i<labelLength; i++) {
            labels4[i] = new MultiLabel();
            predictions4[i] = new MultiLabel();

            if (i < labelLength/2) {
                labels4[i].addLabel(0);
                labels4[i].addLabel(1);
                predictions4[i].addLabel(1);
                predictions4[i].addLabel(0);
            } else {
                labels4[i].addLabel(0);
                predictions4[i].addLabel(1);
            }
        }
        System.out.println("Expected: (value=0.5) - Output: " + Accuracy.accuracy(labels4, predictions4));
    }
}
