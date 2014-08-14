package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.io.*;
import java.util.*;

/**
 * Created by chengli on 8/14/14.
 */
public class LKTreeBoost implements Classifier,ProbabilityEstimator,Serializable {
    private static final long serialVersionUID = 1L;
    /**
     * trees.get(k).get(i) is the ith tree for class k
     */
    private List<List<RegressionTree>> trees;
    private int numClasses;
    private transient LKTBTrainer lktbTrainer;

    public LKTreeBoost(int numClasses) {
        this.numClasses = numClasses;
        this.trees = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<RegressionTree> treesClassK  = new ArrayList<>();
            this.trees.add(treesClassK);
        }
    }


    /**
     * to start/resume training, set train config
     * @param lktbConfig
     */
    public void setTrainConfig(LKTBConfig lktbConfig) {
        if (lktbConfig.getNumClasses()!=this.numClasses){
            throw new RuntimeException("number of classes given in the config does not match number of classes in LKTB");
        }
        this.lktbTrainer = new LKTBTrainer(lktbConfig,this.trees);
    }

    /**
     * default boosting method should follow this order
     * @throws Exception
     */
    public void boostOneRound() throws Exception {
        if (this.lktbTrainer==null){
            throw new RuntimeException("set train config first");
        }
        this.calGradients();
        //for non-standard experiments
        //we can do something here
        //for example, we can add more columns and call setActiveFeatures() here
        this.fitTrees();
    }

    /**
     * parallel by class
     */
    public void calGradients(){
        this.lktbTrainer.calGradients();
    }

    public void fitTrees() throws Exception{
        for (int k=0;k<this.numClasses;k++){
            /**
             * parallel by feature
             */
            RegressionTree regressionTree = this.lktbTrainer.fitClassK(k);
            this.addTree(regressionTree,k);
            /**
             * parallel by data
             */
            this.lktbTrainer.updateStagedScores(regressionTree,k);
        }

        /**
         * parallel by data
         */
        this.lktbTrainer.updateClassProbs();
    }

    /**
     * reset activeDataPoints for later rounds
     * @param activeDataPoints
     */
    public void setActiveDataPoints(int[] activeDataPoints){
        this.lktbTrainer.setActiveDataPoints(activeDataPoints);
    }

    /**
     * reset activeFeatures for later rounds
     * @param activeFeatures
     */
    public void setActiveFeatures(int[] activeFeatures){
        this.lktbTrainer.setActiveFeatures(activeFeatures);
    }

    /**
     * predict the class label
     * @param featureRow
     * @return the class that gives the max class score F
     */
    public int predict(FeatureRow featureRow){
        double maxScore = this.predictClassScore(featureRow, 0);
        int predictedClass = 0;
        for (int k=1;k<this.numClasses;k++){
            double scoreClassK = this.predictClassScore(featureRow, k);
            if (scoreClassK > maxScore){
                maxScore = scoreClassK;
                predictedClass = k;
            }
        }
        return predictedClass;
    }

    /**
     * for external usage
     * @param k
     * @return
     */
    public double[] getGradient(int k){
        return this.lktbTrainer.getGradient(k);
    }



    void addTree(RegressionTree regressionTree, int k){
        this.trees.get(k).add(regressionTree);
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
//        this.updateClassProbs();
//    }





    /**
     *
     * @param featureRow
     * @param k class index
     * @return
     */
    public double predictClassScore(FeatureRow featureRow, int k){
        List<RegressionTree> treesClassK = this.trees.get(k);
        double score = 0;
        for (RegressionTree regressionTree: treesClassK){
            score += regressionTree.predict(featureRow);
        }
        return score;
    }

    public double[] predictClassScores(FeatureRow featureRow){
        double[] scoreVector = new double[this.numClasses];
        for (int k=0;k<this.numClasses;k++){
            scoreVector[k] = this.predictClassScore(featureRow,k);
        }
        return scoreVector;
    }

    public double[] predictClassProbs(FeatureRow featureRow){
        double[] scoreVector = this.predictClassScores(featureRow);
        double[] probVector = new double[this.numClasses];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numClasses;k++){
            double logNominator = scoreVector[k];
            double pro = Math.exp(logNominator-logDenominator);
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
    public RegressionTree getTree(int round, int k){
        return this.trees.get(k).get(round);

    }

    public List<RegressionTree> getTrees(int k){
        return this.trees.get(k);
    }

    //TODO FIX THIS
//    public String showTreesClassK(int k){
//        List<Feature> features = this.dataSet.getFeatures();
//        StringBuilder sb = new StringBuilder();
//        List<RegressionTree> trees = this.trees.get(k);
//        int i=0;
//        for (RegressionTree regressionTree: trees){
//            sb.append("tree "+i+":\n");
//            sb.append(regressionTree.display(features));
//            i += 1;
//        }
//
//        return sb.toString();
//    }

//    /**
//     *
//     * @param k class index
//     * @param features
//     * @return
//     */
//    public Set<String> getSkipNgramNames(int k,List<Feature> features){
//        Set<String> set = new HashSet<String>();
//        for (RegressionTree regressionTree:this.trees.get(k)){
//            Set<String> names = regressionTree.getSkipNgramNames(features);
//            set.addAll(names);
//        }
//        return set;
//    }

    /**sorted
     * only for desicion stump
     * threshold for N
     */
//    public List<String> getTopNgrams(int k,int threshold, List<Feature> features){
//        Map<String,Double> map = new HashMap<String, Double>();
//        for (RegressionTree regressionTree:this.trees.get(k)){
//            if (regressionTree.getNumLeaves()>=2){
//                String featureName = regressionTree.getRootFeatureName(features);
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
     * serialize to file
     * @param lkTreeBoost
     * @param file
     * @throws Exception
     */
    public static void serialize(LKTreeBoost lkTreeBoost, File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(lkTreeBoost);
        }
    }

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

    /**
     * return decision process for class k, sorted by decreasing score absolute values
     * @param featureRow
     * @param features
     * @param k class index
     * @param top only return top decisions
     * @return
     */
//    public String getDecisionProcess(float [] featureRow,List<Feature> features,
//                                     int k, int top){
//        List<DecisionProcess> decisions = new ArrayList<DecisionProcess>();
//        for (int i=0;i<this.getTrees(k).size();i++){
//            RegressionTree regressionTree = this.getTrees(k).get(i);
//            DecisionProcess decisionProcess = regressionTree.getDecisionProcess(featureRow,features);
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
//     * @param featureRow
//     * @param featureNames
//     * @param k class index
//     * @param top only return top decisions
//     * @return
//     */
//    public String getDecisionProcess2(float [] featureRow,List<String> featureNames,
//                                     int k, int top){
//        List<Feature> features = featureNames.stream().map(Feature::new).collect(Collectors.toList());
//        return getDecisionProcess(featureRow,features,k,top);
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
//        RegressionTree regressionTree = this.trees.get(k).get(0);
//        this.trees.get(k).remove(0);
//
//        //update stagedScore of class k
//        for (int i=0;i<this.numDataPoints;i++){
//            float[] featureRow = dataSet.getFeatureRow(i);
//            double prediction = regressionTree.predict(featureRow);
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



}
