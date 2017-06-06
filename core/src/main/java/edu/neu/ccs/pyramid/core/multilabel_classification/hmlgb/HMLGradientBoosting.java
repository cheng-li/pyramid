package edu.neu.ccs.pyramid.core.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * hierarchical multi-label gradient boosting
 * Created by chengli on 9/27/14.
 */
public class HMLGradientBoosting implements MultiLabelClassifier, MultiLabelClassifier.ClassScoreEstimator {
    private static final long serialVersionUID = 2L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    public HMLGradientBoosting(int numClasses, List<MultiLabel> assignments) {
        this.numClasses = numClasses;
        this.assignments = assignments;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
    }

    public int getNumClasses() {
        return numClasses;
    }


    void addRegressor(Regressor regressor, int k){
        this.regressors.get(k).add(regressor);
    }

    List<MultiLabel> getAssignments() {
        return assignments;
    }

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

    double calAssignmentScore(MultiLabel assignment, double[] classScores){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += classScores[label];
        }
        return score;
    }


    public List<Regressor> getRegressors(int k){
        return this.regressors.get(k);
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

    public void serialize(String file) throws Exception{
        serialize(new File(file));
    }

    /**
     * serialize to file
     * @param file
     * @throws Exception
     */
    public void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    public static HMLGradientBoosting deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }

    /**
     * de-serialize from file
     * @param file
     * @return
     * @throws Exception
     */
    public static HMLGradientBoosting deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            HMLGradientBoosting boosting = (HMLGradientBoosting)objectInputStream.readObject();
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
}
