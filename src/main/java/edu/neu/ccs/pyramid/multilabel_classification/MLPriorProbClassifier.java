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
    private double[] assignmentProbs;
    private List<MultiLabel> assignments;

    public MLPriorProbClassifier(int numClasses, List<MultiLabel> assignments) {
        this.numClasses = numClasses;

        this.assignments = assignments;
        this.classProbs = new double[numClasses];
        this.assignmentProbs = new double[assignments.size()];
    }

    public void fit(MultiLabelClfDataSet dataSet){
        for (int a=0;a<this.assignments.size();a++){
            MultiLabel assignment = this.assignments.get(a);
            this.assignmentProbs[a] = this.calAssignmentProb(dataSet,assignment);
        }
        for (int a=0;a<this.assignments.size();a++){
            MultiLabel assignment = this.assignments.get(a);
            for (Integer label: assignment.getMatchedLabels()){
                classProbs[label] += this.assignmentProbs[a];
            }
        }
    }

    public double[] getClassProbs() {
        return classProbs;
    }

    public double[] getAssignmentProbs() {
        return assignmentProbs;
    }

    private double calAssignmentProb(MultiLabelClfDataSet dataSet, MultiLabel assignment){
        double  numDataPoints = dataSet.getNumDataPoints();
        double count = 0;
        for (MultiLabel multiLabel: dataSet.getMultiLabels()){
            if (MultiLabel.equivalent(assignment,multiLabel)){
                count += 1;
            }
        }
        return count/numDataPoints;
    }

    @Override
    public String toString() {
        return "MLPriorProbClassifier{" +
                "numClasses=" + numClasses +
                ", classProbs=" + Arrays.toString(classProbs) +
                ", assignmentProbs=" + Arrays.toString(assignmentProbs) +
                ", assignments=" + assignments +
                '}';
    }
}
