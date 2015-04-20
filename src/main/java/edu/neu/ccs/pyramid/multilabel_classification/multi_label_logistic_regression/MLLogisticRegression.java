package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
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
public class MLLogisticRegression implements MultiLabelClassifier, MultiLabelClassifier.ClassScoreEstimator {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFeatures;
    private Weights weights;
    private boolean featureExtraction = false;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;


    public MLLogisticRegression(int numClasses, int numFeatures, List<MultiLabel> assignments) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
        this.assignments = assignments;
    }

    public MLLogisticRegression(int numClasses, int numFeatures,
                                List<MultiLabel> assignments, Vector weightVector) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures, weightVector);
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
        double[] classeScores = predictClassScores(vector);
        for (MultiLabel assignment: this.assignments){
            double score = this.calAssignmentScore(assignment,classeScores);
            if (score > maxScore){
                maxScore = score;
                prediction = assignment;
            }
        }
        return prediction;
    }

    public double predictClassScore(Vector dataPoint, int k){
        double score = 0;
        score += this.weights.getBiasForClass(k);
        score += this.weights.getWeightsWithoutBiasForClass(k).dot(dataPoint);
        return score;
    }

    public double[] predictClassScores(Vector dataPoint){
        double[] scores = new double[this.numClasses];
        for (int k=0;k<this.numClasses;k++){
            scores[k] = predictClassScore(dataPoint, k);
        }
        return scores;
    }


    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        if (!this.assignments.contains(assignment)){
            return 0;
        }
        double[] classScores = predictClassScores(vector);
        double[] assignmentScores = new double[this.assignments.size()];
        for (int i=0;i<assignments.size();i++){
            assignmentScores[i] = calAssignmentScore(assignments.get(i),classScores);
        }
        double logNumerator = calAssignmentScore(assignment,classScores);
        double logDenominator = MathUtil.logSumExp(assignmentScores);
        double pro = Math.exp(logNumerator-logDenominator);
        return pro;
    }


    /**
     * for legal assignments
     * @param vector
     * @return
     */
    double[] predictAssignmentProbs(Vector vector){
        double[] classScores = predictClassScores(vector);
        double[] assignmentScores = new double[this.assignments.size()];
        for (int i=0;i<assignments.size();i++){
            assignmentScores[i] = calAssignmentScore(assignments.get(i),classScores);
        }
        double logDenominator = MathUtil.logSumExp(assignmentScores);
        double[] assignmentProbs = new double[this.assignments.size()];
        for (int i=0;i<assignments.size();i++){
            double logNumerator = assignmentScores[i];
            double pro = Math.exp(logNumerator-logDenominator);
            assignmentProbs[i] = pro;
        }
        return assignmentProbs;
    }


    /**
     * expensive operation
     * @param vector
     * @return
     */
    public double[] predictClassProbs(Vector vector){
        double[] assignmentProbs = predictAssignmentProbs(vector);
        double[] classProbs = new double[numClasses];
        for (int a=0;a<assignments.size();a++){
            MultiLabel assignment = assignments.get(a);
            double prob = assignmentProbs[a];
            for (Integer label:assignment.getMatchedLabels()){
                double oldProb = classProbs[label];
                classProbs[label] = oldProb + prob;
            }
        }
        return classProbs;
    }

    public double predictClassProb(Vector vector, int classIndex){
        return predictClassProbs(vector)[classIndex];
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
            double logNumerator = assignmentScores[a];
            double pro = Math.exp(logNumerator-logDenominator);
            assignmentProbs[a]=pro;
        }
        return assignmentProbs;
    }


    double logLikelihood(Vector vector, MultiLabel multiLabel){
        double[] classScores = predictClassScores(vector);
        int numAssignments = assignments.size();
        double[] assignmentScores = new double[numAssignments];
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            assignmentScores[a] = this.calAssignmentScore(assignment, classScores);
        }
        double logDenominator = MathUtil.logSumExp(assignmentScores);

        double logNumerator = this.calAssignmentScore(multiLabel, classScores);
        return logNumerator-logDenominator;
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

    public FeatureList getFeatureList() {
        return featureList;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
