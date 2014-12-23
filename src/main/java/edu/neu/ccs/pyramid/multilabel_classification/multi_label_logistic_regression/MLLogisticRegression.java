package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 12/23/14.
 */
public class MLLogisticRegression implements MultiLabelClassifier{
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFeatures;
    private Weights weights;
    private boolean featureExtraction = false;
    private String[] featureNames;
    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;


    public MLLogisticRegression(int numClasses, int numFeatures, List<MultiLabel> assignments) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
        this.featureNames = new String[numFeatures];
        this.assignments = assignments;
    }

    public MLLogisticRegression(int numClasses, int numFeatures,
                                List<MultiLabel> assignments, Vector weightVector) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures, weightVector);
        this.featureNames = new String[numFeatures];
        this.assignments = assignments;
    }

    public List<MultiLabel> getAssignments() {
        return assignments;
    }

    public boolean featureExtraction() {
        return featureExtraction;
    }

    public void setFeatureExtraction(boolean featureExtraction) {
        this.featureExtraction = featureExtraction;
    }

    public Weights getWeights() {
        return weights;
    }


    public int getNumFeatures() {
        return numFeatures;
    }

    public int getNumClasses(){
        return this.numClasses;
    }

    @Override
    public MultiLabel predict(Vector vector){
        double maxScore = Double.NEGATIVE_INFINITY;
        MultiLabel prediction = null;
        double[] classeScores = calClassScores(vector);
        for (MultiLabel assignment: this.assignments){
            double score = this.calAssignmentScore(assignment,classeScores);
            if (score > maxScore){
                maxScore = score;
                prediction = assignment;
            }
        }
        return prediction;
    }

    public double calClassScore(Vector dataPoint, int k){
        double score = 0;
        score += this.weights.getBiasForClass(k);
        score += this.weights.getWeightsWithoutBiasForClass(k).dot(dataPoint);
        return score;
    }

    public double[] calClassScores(Vector dataPoint){
        double[] scores = new double[this.numClasses];
        for (int k=0;k<this.numClasses;k++){
            scores[k] = calClassScore(dataPoint, k);
        }
        return scores;
    }

    double calAssignmentScore(MultiLabel assignment, double[] classScores){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += classScores[label];
        }
        return score;
    }

    double[] calAssignmentScores(double[] classScores){
        double[] scores = new double[assignments.size()];
        for (int a=0;a<scores.length;a++){
            scores[a] = calAssignmentScore(assignments.get(a),classScores);
        }
        return scores;
    }


    double[] calAssignmentProbs(double[] assignmentScores){
        int numAssignments = assignments.size();
        double[] assignmentProbs = new double[numAssignments];
        double logDenominator = MathUtil.logSumExp(assignmentScores);
        for (int a=0;a<numAssignments;a++){
            double logNominator = assignmentScores[a];
            double pro = Math.exp(logNominator-logDenominator);
            assignmentProbs[a]=pro;
        }
        return assignmentProbs;
    }


    double logLikelihood(Vector vector, MultiLabel multiLabel){
        double[] classScores = calClassScores(vector);
        int numAssignments = assignments.size();
        double[] assignmentScores = new double[numAssignments];
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            assignmentScores[a] = this.calAssignmentScore(assignment, classScores);
        }
        double logDenominator = MathUtil.logSumExp(assignmentScores);

        double logNominator = this.calAssignmentScore(multiLabel, classScores);
        return logNominator-logDenominator;
    }


    double dataSetLogLikelihood(MultiLabelClfDataSet dataSet){
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i),multiLabels[i]))
                .sum();
    }




    public static MLLogisticRegression deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            return (MLLogisticRegression)objectInputStream.readObject();
        }
    }

    void setFeatureName(int featureIndex, String featureName){
        featureNames[featureIndex] = featureName;
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    double[] calClassProbs(double[] assignmentProbs){
        double[] classProbs = new double[numClasses];
        int numAssignments = assignments.size();
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            double prob = assignmentProbs[a];
            for (Integer label:assignment.getMatchedLabels()){
                classProbs[label] += prob;
            }
        }
        return classProbs;
    }
    
}
