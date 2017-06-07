package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * calculate prior probability for each assignment and each class
 * Created by chengli on 9/28/14.
 */
public class MLPriorProbClassifier implements Serializable{
    private static final long serialVersionUID = 1L;

    private int numClasses;
    private double[] classProbs;

    public MLPriorProbClassifier(int numClasses) {
        this.numClasses = numClasses;

        this.classProbs = new double[numClasses];
    }

    public void fit(MultiLabelClfDataSet dataSet){
        if (dataSet.getNumClasses()!=this.numClasses){
            throw new IllegalArgumentException("dataSet.getNumClasses()!=this.numClasses");
        }
        int[] counts = new int[this.numClasses];
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        for (MultiLabel multiLabel:multiLabels){
            for (int matchedClass: multiLabel.getMatchedLabels()){
                counts[matchedClass] += 1;
            }
        }
        for (int k=0;k<this.numClasses;k++){
            this.classProbs[k] = ((double)counts[k])/dataSet.getNumDataPoints();
        }
    }

    public double[] getClassProbs() {
        return classProbs;
    }

    @Override
    public String toString() {
        return "MLPriorProbClassifier{" +
                "numClasses=" + numClasses +
                ", classProbs=" + Arrays.toString(classProbs) +
                '}';
    }
}
