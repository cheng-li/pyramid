package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * gradient boosting for independent labels
 * the training part does not consider any label relations
 * the prediction part can consider label relations
 * Created by chengli on 10/8/14.
 */
public class IMLGradientBoosting implements MultiLabelClassifier.ClassScoreEstimator, MultiLabelClassifier.ClassProbEstimator {
    private static final long serialVersionUID = 2L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    /**
     * legal assignments of labels, optional
     */
    private List<MultiLabel> assignments;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;
    private PredictFashion predictFashion = PredictFashion.INDEPENDENT;

    public IMLGradientBoosting(int numClasses) {
        this.numClasses = numClasses;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
    }

    public int getNumClasses() {
        return numClasses;
    }

    public List<MultiLabel> getAssignments() {
        return assignments;
    }

    public void setAssignments(List<MultiLabel> assignments) {
        this.assignments = assignments;
    }

    void addRegressor(Regressor regressor, int k){
        this.regressors.get(k).add(regressor);
    }


    public PredictFashion getPredictFashion() {
        return predictFashion;
    }

    public void setPredictFashion(PredictFashion predictFashion) {
        this.predictFashion = predictFashion;
    }

    /**
     * independent do prediction without any constraint;
     * if crf, only consider these assignments
     * @param vector
     * @return
     */
    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = null;
        switch (this.predictFashion){
            case INDEPENDENT:
                prediction = predictWithoutConstraints(vector);
                break;
            case CRF:
                prediction = predictWithConstraints(vector);
                break;
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
        if (this.assignments==null){
            throw new RuntimeException("CRF is used but legal assignments is not specified!");
        }
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

    double calAssignmentScore(MultiLabel assignment, double[] classScores){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += classScores[label];
        }
        return score;
    }

    /**
     *
     * @param vector
     * @param k class index
     * @return
     */
    public double predictClassScore(Vector vector, int k){
        List<Regressor> regressorsClassK = this.regressors.get(k);
        double score = 0;
        for (Regressor regressor: regressorsClassK){
            score += regressor.predict(vector);
        }
        return score;
    }

    public double[] predictClassScores(Vector vector){
        int numClasses = this.numClasses;
        double[] scores = new double[numClasses];
        for (int k=0;k<numClasses;k++){
            scores[k] = this.predictClassScore(vector, k);
        }
        return scores;
    }


    public List<Regressor> getRegressors(int k){
        return this.regressors.get(k);
    }


    //todo think about this when having assignments, maybe doesn't matter much
    public double predictClassProb(Vector vector, int classIndex){
        double score = predictClassScore(vector,classIndex);
        double logNumerator = score;
        double[] scores = new double[2];
        scores[0] = 0;
        scores[1] = score;
        double logDenominator = MathUtil.logSumExp(scores);
        double pro = Math.exp(logNumerator-logDenominator);
        return pro;
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        return IntStream.range(0,numClasses)
                .mapToDouble(k -> predictClassProb(vector,k)).toArray();
    }

    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        if (assignment.outOfBound(this.numClasses)){
            return 0;
        }
        double prob = 0;
        switch (this.predictFashion){
            case INDEPENDENT:
                prob = predictAssignmentProbWithoutConstraint(vector,assignment);
                break;
            case CRF:
                prob = predictAssignmentProbWithConstraint(vector,assignment);
                break;
        }
        return prob;
    }

    double predictAssignmentProbWithConstraint(Vector vector, MultiLabel assignment){
        if (this.assignments==null){
            throw new RuntimeException("CRF is used but legal assignments is not specified!");
        }
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



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int k=0;k<this.numClasses;k++){
            sb.append("for class ").append(k).append("\n");
            List<Regressor> trees = this.getRegressors(k);
            for (int i=0;i<trees.size();i++){
                sb.append("tree ").append(i).append(":");
                sb.append(trees.get(i).toString());
            }
        }
        return sb.toString();
    }



    public static IMLGradientBoosting deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }

    /**
     * de-serialize from file
     * @param file
     * @return
     * @throws Exception
     */
    public static IMLGradientBoosting deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            IMLGradientBoosting boosting = (IMLGradientBoosting)objectInputStream.readObject();
            return boosting;
        }
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }


    public static enum PredictFashion {
        CRF, INDEPENDENT
    }
}
