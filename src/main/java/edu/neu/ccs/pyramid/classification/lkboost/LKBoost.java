package edu.neu.ccs.pyramid.classification.lkboost;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.Ensemble;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;


/**
 * Created by chengli on 8/14/14.
 */
public class LKBoost extends GradientBoosting implements Classifier.ProbabilityEstimator, Classifier.ScoreEstimator{
    private static final long serialVersionUID = 4L;
    private int numClasses;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    public LKBoost(int numClasses) {
        super(numClasses);
        this.numClasses = numClasses;
    }

    /**
     * predict the class label
     * @param vector
     * @return the class that gives the max class score F
     */
    public int predict(Vector vector){
        double[] scores = predictClassScores(vector);
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = 0;
        for (int k=0;k<this.numClasses;k++){
            double scoreClassK = scores[k];
            if (scoreClassK > maxScore){
                maxScore = scoreClassK;
                predictedClass = k;
            }
        }
        return predictedClass;
    }

    public int getNumClasses() {
        return this.numClasses;
    }

    //    /**
//     * remove first n tree for all classes
//     * calibrate the scores and probabilities
//     */
//    public void removeFirstNTrees(int n){
//        for (int i=0;i<n;i++){
//            this.removeFirstTree();
//        }
//        this.updateProbabilityMatrix();
//    }


    /**
     *
     * @param vector
     * @param k class index
     * @return
     */
    public double predictClassScore(Vector vector, int k){
        return getEnsemble(k).score(vector);
    }

    public double[] predictClassScores(Vector vector){
        double[] scoreVector = new double[this.numClasses];
        for (int k=0;k<this.numClasses;k++){
            scoreVector[k] = this.predictClassScore(vector,k);
        }
        return scoreVector;
    }


    public double[] predictClassProbs(Vector vector){
        double[] scoreVector = this.predictClassScores(vector);
        double[] probVector = new double[this.numClasses];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numClasses;k++){
            double logNumerator = scoreVector[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }

    @Override
    public double[] predictLogClassProbs(Vector vector) {
        double[] scoreVector = this.predictClassScores(vector);
        double[] logProbVector = new double[this.numClasses];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numClasses;k++){
            double logNumerator = scoreVector[k];
            double logPro = logNumerator-logDenominator;
            logProbVector[k]= logPro;
        }
        return logProbVector;
    }




    /**
     * de-serialize from file
     * @param file
     * @return
     * @throws Exception
     */
    public static LKBoost deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            LKBoost lkBoost = (LKBoost)objectInputStream.readObject();
            return lkBoost;
        }
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int k=0;k<this.numClasses;k++){
            sb.append("for class ").append(k).append("\n");
            Ensemble trees = this.getEnsemble(k);
            for (int i=0;i<trees.getRegressors().size();i++){
                sb.append("tree ").append(i).append(":");
                sb.append(trees.get(i).toString());
            }
        }
        return sb.toString();
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
