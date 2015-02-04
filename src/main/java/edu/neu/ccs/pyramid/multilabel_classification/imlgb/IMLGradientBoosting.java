package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MLPriorProbClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * gradient boosting for independent labels
 * the training part does not consider any label relations
 * the prediction part can consider label relations
 * Created by chengli on 10/8/14.
 */
public class IMLGradientBoosting implements MultiLabelClassifier{
    private static final long serialVersionUID = 2L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    /**
     * legal assignments of labels, optional
     */
    private List<MultiLabel> assignments;

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
            double score = this.calClassScore(vector,k);
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
        double[] classScores = calClassScores(vector);
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
    public double calClassScore(Vector vector, int k){
        List<Regressor> regressorsClassK = this.regressors.get(k);
        double score = 0;
        for (Regressor regressor: regressorsClassK){
            score += regressor.predict(vector);
        }
        return score;
    }

    double[] calClassScores(Vector vector){
        int numClasses = this.numClasses;
        double[] scores = new double[numClasses];
        for (int k=0;k<numClasses;k++){
            scores[k] = this.calClassScore(vector,k);
        }
        return scores;
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

}
