package edu.neu.ccs.pyramid.multilabel_classification.imllr;

import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 5/15/15.
 */
public class IMLLogisticRegression implements MultiLabelClassifier, MultiLabelClassifier.ClassScoreEstimator{
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFeatures;
    private Weights weights;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;


    public IMLLogisticRegression(int numClasses, int numFeatures, List<MultiLabel> assignments) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
        this.assignments = assignments;
    }

    public IMLLogisticRegression(int numClasses, int numFeatures,
                                List<MultiLabel> assignments, Vector weightVector) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures, weightVector);
        this.assignments = assignments;
    }

    public List<MultiLabel> getAssignments() {
        return assignments;
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

    /**
     * if legal assignments are not present, do prediction without any constraint;
     * if legal assignments are present, only consider these assignments
     * @param vector
     * @return
     */
    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction;
        if (this.assignments!=null){
            prediction = predictWithConstraints(vector);
        } else {
            prediction = predictWithoutConstraints(vector);
        }
        return prediction;
    }

    /**
     * do prediction without any constraint
     * @param vector
     * @return
     */
    private MultiLabel predictWithoutConstraints(Vector vector){
        MultiLabel prediction = new MultiLabel();
        for (int k=0;k<numClasses;k++){
            double score = this.predictClassScore(vector, k);
            if (score > 0){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }

    /**
     * only consider these assignments
     * @param vector
     * @return
     */
    private MultiLabel predictWithConstraints(Vector vector){
        double maxScore = Double.NEGATIVE_INFINITY;
        MultiLabel prediction = null;
        double[] classScores = predictClassScores(vector);
        for (MultiLabel assignment: this.assignments){
            double score = this.calAssignmentScore(assignment,classScores);
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
        if (assignment.outOfBound(this.numClasses)){
            return 0;
        }
        if (this.assignments!=null){
            return predictAssignmentProbWithConstraint(vector,assignment);
        } else {
            return predictAssignmentProbWithoutConstraint(vector,assignment);
        }
    }

    double predictAssignmentProbWithConstraint(Vector vector, MultiLabel assignment){
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

    double predictAssignmentProbWithoutConstraint(Vector vector, MultiLabel assignment){
        double[] classScores = predictClassScores(vector);
        double logProb = 0;
        for (int k=0;k<numClasses;k++){
            double logNumerator = 0;
            if (assignment.matchClass(k)){
                logNumerator = classScores[k];
            }
            double[] scores = new double[2];
            scores[0] = 0;
            scores[1] = classScores[k];
            double logDenominator = MathUtil.logSumExp(scores);

            logProb += logNumerator;
            logProb -= logDenominator;
        }

        return Math.exp(logProb);
    }


    public double predictClassProb(Vector vector, int classIndex){
        double score = predictClassScore(vector, classIndex);
        double logNumerator = score;
        double[] scores = new double[2];
        scores[0] = 0;
        scores[1] = score;
        double logDenominator = MathUtil.logSumExp(scores);
        double pro = Math.exp(logNumerator-logDenominator);
        return pro;
    }


    public double[] predictClassProbs(Vector vector) {
        return IntStream.range(0,numClasses)
                .mapToDouble(k -> predictClassProb(vector,k)).toArray();
    }


    double calAssignmentScore(MultiLabel assignment, double[] classScores){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += classScores[label];
        }
        return score;
    }




    // no constraints in training
    double logLikelihood(Vector vector, MultiLabel assignment){
        double[] classScores = predictClassScores(vector);
        double logProb = 0;
        for (int k=0;k<numClasses;k++){
            double logNumerator = 0;
            if (assignment.matchClass(k)){
                logNumerator = classScores[k];
            }
            double[] scores = new double[2];
            scores[0] = 0;
            scores[1] = classScores[k];
            double logDenominator = MathUtil.logSumExp(scores);

            logProb += logNumerator;
            logProb -= logDenominator;
        }

        return logProb;
    }


    double dataSetLogLikelihood(MultiLabelClfDataSet dataSet){
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i),multiLabels[i]))
                .sum();
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
