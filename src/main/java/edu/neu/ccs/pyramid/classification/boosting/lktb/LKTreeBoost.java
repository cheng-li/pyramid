package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by chengli on 8/14/14.
 */
public class LKTreeBoost implements Classifier.ProbabilityEstimator, Classifier.ScoreEstimator{
    private static final long serialVersionUID = 3L;
    /**
     * regressors.get(k).get(i) is the ith regressor for class k
     */
    private List<List<Regressor>> regressors;
    private int numClasses;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    public LKTreeBoost(int numClasses) {
        this.numClasses = numClasses;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
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
        List<Regressor> regressorsClassK = this.regressors.get(k);
        double score = 0;
        for (Regressor regressor: regressorsClassK){
            score += regressor.predict(vector);
        }
        return score;
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

//    /**
//     *
//     * @param norm "L1" or "L2"
//     * @return
//     */
//    public double[] getWeights(String norm){
//        boolean condition = (norm.equals("L1") || norm.equals("L2") || norm.equals("MAX"));
//        if (!condition){
//            throw new IllegalArgumentException("pick the right norm") ;
//        }
//        double[] weights = new double[this.numDataPoints];
//        if (norm.equals("L1")){
//            for (int i=0;i<this.numDataPoints;i++){
//                double[] gradients = new double[this.numClasses];
//                for (int k=0;k<this.numClasses;k++){
//                    gradients[k] = this.classLabels[k][i]- this.classProbabilities[k][i];
//                }
//                weights[i] = MathUtil.l1Norm(gradients);
//            }
//        } else if (norm.equals("L2")){
//            for (int i=0;i<this.numDataPoints;i++){
//                double[] gradients = new double[this.numClasses];
//                for (int k=0;k<this.numClasses;k++){
//                    gradients[k] = this.classLabels[k][i]- this.classProbabilities[k][i];
//                }
//                weights[i] = MathUtil.l2Norm(gradients);
//            }
//        } else if (norm.equals("MAX")){
//            for (int i=0;i<this.numDataPoints;i++){
//                double[] gradients = new double[this.numClasses];
//                for (int k=0;k<this.numClasses;k++){
//                    gradients[k] = this.classLabels[k][i]- this.classProbabilities[k][i];
//                }
//                weights[i] = MathUtil.maxNorm(gradients);
//            }
//        }
//        return weights;
//    }

    /**
     *
     * @param round
     * @param k class index
     * @return
     */
    public Regressor getRegressor(int round, int k){
        return this.regressors.get(k).get(round);

    }

    public List<Regressor> getRegressors(int k){
        return this.regressors.get(k);
    }

    void addRegressor(Regressor regressor, int k){
        this.regressors.get(k).add(regressor);
    }

    //TODO FIX THIS
//    public String showTreesClassK(int k){
//        List<Feature> featureList = this.dataSet.getFeatureList();
//        StringBuilder sb = new StringBuilder();
//        List<RegressionTree> regressors = this.regressors.get(k);
//        int i=0;
//        for (RegressionTree regressionTree: regressors){
//            sb.append("tree "+i+":\n");
//            sb.append(regressionTree.display(featureList));
//            i += 1;
//        }
//
//        return sb.toString();
//    }

//    /**
//     *
//     * @param k class index
//     * @param featureList
//     * @return
//     */
//    public Set<String> getSkipNgramNames(int k,List<Feature> featureList){
//        Set<String> set = new HashSet<String>();
//        for (RegressionTree regressionTree:this.regressors.get(k)){
//            Set<String> names = regressionTree.getSkipNgramNames(featureList);
//            set.addAll(names);
//        }
//        return set;
//    }

    /**sorted
     * only for desicion stump
     * threshold for N
     */
//    public List<String> getTopNgrams(int k,int threshold, List<Feature> featureList){
//        Map<String,Double> map = new HashMap<String, Double>();
//        for (RegressionTree regressionTree:this.regressors.get(k)){
//            if (regressionTree.getNumLeaves()>=2){
//                String featureName = regressionTree.getRootFeatureName(featureList);
//                Ngram ngram = new Ngram(featureName);
//                if (ngram.getNumTerms()>=threshold){
//                    double score = regressionTree.getRootRightOutput();
//                    Double oldScore = map.get(featureName);
//                    if (oldScore==null){
//                        map.put(featureName,0.0);
//                        oldScore = map.get(featureName);
//                    }
//                    double newScore = oldScore + score;
//                    map.put(featureName,newScore);
//                }
//            }
//
//        }
////        System.out.println("skip ngrams score:");
////        System.out.println(map);
//        List<String> topNgrams = (new MapSort<String>()).keysDescending(map);
//        return topNgrams;
//    }


    /**
     * de-serialize from file
     * @param file
     * @return
     * @throws Exception
     */
    public static LKTreeBoost deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            LKTreeBoost lkTreeBoost = (LKTreeBoost)objectInputStream.readObject();
            return lkTreeBoost;
        }
    }

//    public String getDecisionProcess(float [] vector,List<Feature> featureList,
//                                     int k, int top){
//        List<DecisionProcess> decisions = new ArrayList<DecisionProcess>();
//        for (int i=0;i<this.getRegressors(k).size();i++){
//            RegressionTree regressionTree = this.getRegressors(k).get(i);
//            DecisionProcess decisionProcess = regressionTree.getDecisionProcess(vector,featureList);
//            decisionProcess.setTreeIndex(i);
//            decisions.add(decisionProcess);
//        }
//
//        Collections.sort(decisions,DecisionProcess.ScoreDescComprator);
//
//        StringBuilder stringBuilder = new StringBuilder();
//        for (int i=0;i<Math.min(top,decisions.size());i++){
//            DecisionProcess decisionProcess = decisions.get(i);
//            stringBuilder.append(decisionProcess.toString());
//            stringBuilder.append("\n");
//        }
//        return stringBuilder.toString();
//    }

//    /**
//     * return decision process for class k, sorted by decreasing score absolute values
//     * @param vector
//     * @param featureNames
//     * @param k class index
//     * @param top only return top decisions
//     * @return
//     */
//    public String getDecisionProcess2(float [] vector,List<String> featureNames,
//                                     int k, int top){
//        List<Feature> featureList = featureNames.stream().map(Feature::new).collect(Collectors.toList());
//        return getDecisionProcess(vector,featureList,k,top);
//    }


//    /**
//     * sort by ascending prob(true label)
//     * (descending 1-prob)
//     * @param maxSize max number of feature points to be put into focus set
//     * @return
//     */
//    public FocusSet getFocusSet(int maxSize){
//        FocusSet focusSet = new FocusSet(this.numClasses);
//        double[] probs = new double[this.numDataPoints];
//        for (int i=0;i<this.numDataPoints;i++){
//            int trueLabel = this.trueLabels[i];
//            probs[i] = this.classProbabilities[trueLabel][i];
//        }
//        int[] sortedIndices = Argsort.argSortAscending(probs);
//        for (int j=0;j<maxSize;j++){
//            int dataPointIndex = sortedIndices[j];
//            int trueLabel = this.trueLabels[dataPointIndex];
////            System.out.println(dataPointIndex);
////            System.out.println(probs[dataPointIndex]);
//            focusSet.add(dataPointIndex,trueLabel);
//        }
//        return focusSet;
//    }

    //PRIVATE


//    /**
//     * remove the first tree from class k, calibrate scores
//     * @param k class index
//     */
//    private void removeFirstTree(int k){
//        RegressionTree regressionTree = this.regressors.get(k).get(0);
//        this.regressors.get(k).remove(0);
//
//        //update stagedScore of class k
//        for (int i=0;i<this.numDataPoints;i++){
//            float[] vector = dataSet.getRow(i);
//            double prediction = regressionTree.predict(vector);
//            if (Double.isNaN(prediction)){
//                throw new RuntimeException("prediction is NaN");
//            }
//            this.stagedScore[k][i] -= prediction;
//        }
//    }
//
//    /**
//     * remove the first tree for all classes
//     * calibrate scores
//     */
//    private void removeFirstTree(){
//        for (int k=0;k<this.numClasses;k++){
//            this.removeFirstTree(k);
//        }
//    }

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
