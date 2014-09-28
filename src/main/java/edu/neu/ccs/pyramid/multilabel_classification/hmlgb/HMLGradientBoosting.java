package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.regression.Regressor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/27/14.
 */
public class HMLGradientBoosting implements MultiLabelClassifier, Serializable{
    private static final long serialVersionUID = 1L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;
    private transient HMLGBTrainer trainer;

    public HMLGradientBoosting(int numClasses, List<MultiLabel> assignments) {
        this.numClasses = numClasses;
        this.assignments = assignments;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
    }

    /**
     * to start/resume training, set train config
     * @param config
     */
    public void setTrainConfig(HMLGBConfig config) {
        if (config.getDataSet().getNumClasses()!=this.numClasses){
            throw new RuntimeException("number of classes given in the config does not match number of classes in boosting");
        }
        this.trainer = new HMLGBTrainer(config,this.regressors, this.assignments);
    }

    /**
     * default boosting method should follow this order
     * @throws Exception
     */
    public void boostOneRound()  {
        if (this.trainer==null){
            throw new RuntimeException("set train config first");
        }
        this.calGradients();
        //for non-standard experiments
        //we can do something here
        //for example, we can add more columns and call setActiveFeatures() here
        this.fitRegressors();
    }

    /**
     * for external usage
     * @param k
     * @return
     */
    public double[] getGradients(int k){
        return this.trainer.getGradients(k);
    }

    /**
     * parallel by class
     */
    public void calGradients(){
        this.trainer.updateClassGradientMatrix();
    }

    public void fitRegressors(){
        for (int k=0;k<this.numClasses;k++){
            /**
             * parallel by feature
             */
            Regressor regressor = this.trainer.fitClassK(k);
            this.addRegressor(regressor, k);
            /**
             * parallel by data
             */
            this.trainer.updateStagedClassScores(regressor,k);
        }

        /**
         * parallel by data
         */
        this.trainer.updateAssignmentProbMatrix();
    }

    void addRegressor(Regressor regressor, int k){
        this.regressors.get(k).add(regressor);
    }

    /**
     * reset activeDataPoints for later rounds
     * @param activeDataPoints
     */
    public void setActiveDataPoints(int[] activeDataPoints){
        this.trainer.setActiveDataPoints(activeDataPoints);
    }

    /**
     * reset activeFeatures for later rounds
     * @param activeFeatures
     */
    public void setActiveFeatures(int[] activeFeatures){
        this.trainer.setActiveFeatures(activeFeatures);
    }

    public MultiLabel predict(FeatureRow featureRow){
        double maxScore = Double.NEGATIVE_INFINITY;
        MultiLabel prediction = null;
        for (MultiLabel assignment: this.assignments){
            double score = this.calAssignmentScores(featureRow,assignment);
            if (score > maxScore){
                maxScore = score;
                prediction = assignment;
            }
        }
        return prediction;
    }

    double calAssignmentScores(FeatureRow featureRow, MultiLabel assignment){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += this.calClassScore(featureRow,label);
        }
        return score;
    }

    /**
     *
     * @param featureRow
     * @param k class index
     * @return
     */
    public double calClassScore(FeatureRow featureRow, int k){
        List<Regressor> regressorsClassK = this.regressors.get(k);
        double score = 0;
        for (Regressor regressor: regressorsClassK){
            score += regressor.predict(featureRow);
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


}
