package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.optimization.EarlyStopper;
import edu.neu.ccs.pyramid.ranking.LambdaMART;
import edu.neu.ccs.pyramid.ranking.LambdaMARTOptimizer;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.ls_logistic_boost.LSLogisticBoost;
import edu.neu.ccs.pyramid.regression.ls_logistic_boost.LSLogisticBoostOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.IOException;
import java.util.Arrays;

public class RerankerTrainer {
    private int numLeaves;
    private int numCandidates;
    private double shrinkage;
    private int minDataPerLeaf;
    private int maxIter;
    //"none", "weak", "strong", "xgboost"
    private String monotonicityType;




    public Reranker train(RegDataSet regDataSet, double[] instanceWeights,
                          PredictionFeatureExtractor predictionFeatureExtractor, RegDataSet validation){
        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf).setMonotonicityType(monotonicityType);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, regDataSet, regTreeFactory, instanceWeights, regDataSet.getLabels());
        if (!monotonicityType.equals("none")){
            int[][] mono = new int[1][regDataSet.getNumFeatures()];
            mono[0] = predictionFeatureExtractor.featureMonotonicity();
            optimizer.setMonotonicity(mono);
        }
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE,5);
        LSBoost bestModel = null;
        for (int i = 1; i<=maxIter; i++){
            optimizer.iterate();

            if (i%10==0|| i==maxIter){
                double mse = MSE.mse(lsBoost, validation);
                earlyStopper.add(i,mse);
                if (earlyStopper.getBestIteration()==i){
                    try {
                        bestModel = (LSBoost) Serialization.deepCopy(lsBoost);
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                if (earlyStopper.shouldStop()){
                    break;
                }
            }
        }
//        System.out.println("best iteration = "+earlyStopper.getBestIteration());
        return new Reranker(bestModel, numCandidates,predictionFeatureExtractor);
    }




    public Reranker trainWithSigmoid(RegDataSet regDataSet, double[] instanceWeights,
                                     PredictionFeatureExtractor predictionFeatureExtractor,  RegDataSet validation,
                                     double[] noiseRates0, double[] noiseRates1){
        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf).setMonotonicityType(monotonicityType);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost, regDataSet, regTreeFactory, instanceWeights, regDataSet.getLabels(), noiseRates0, noiseRates1);
//        optimizer.setNoiseRates0(noiseRates0);
//        optimizer.setNoiseRates1(noiseRates1);
        if (!monotonicityType.equals("none")){
            int[][] mono = new int[1][regDataSet.getNumFeatures()];
            mono[0] = predictionFeatureExtractor.featureMonotonicity();
            optimizer.setMonotonicity(mono);
        }
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();

        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE,5);
        LSLogisticBoost bestModel = null;
        for (int i = 1; i<=maxIter; i++){
            optimizer.iterate();

            if (i%10==0){
                double mse = MSE.mse(lsLogisticBoost, validation);
                //todo
//                double trainMse = MSE.mse(lsLogisticBoost, regDataSet);
//                System.out.println("iter="+i+", train mse="+trainMse+" , valid mse="+mse);
                earlyStopper.add(i,mse);
                if (earlyStopper.getBestIteration()==i){
                    try {
                        bestModel = (LSLogisticBoost) Serialization.deepCopy(lsLogisticBoost);
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                if (earlyStopper.shouldStop()){
                    break;
                }
            }
        }
        return new Reranker(bestModel, numCandidates,predictionFeatureExtractor);
    }


    public Reranker trainWithSigmoid(RegDataSet regDataSet, double[] instanceWeights,
                                     PredictionFeatureExtractor predictionFeatureExtractor,  RegDataSet validation){
        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf).setMonotonicityType(monotonicityType);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost, regDataSet, regTreeFactory, instanceWeights, regDataSet.getLabels());
        if (!monotonicityType.equals("none")){
            int[][] mono = new int[1][regDataSet.getNumFeatures()];
            mono[0] = predictionFeatureExtractor.featureMonotonicity();
            optimizer.setMonotonicity(mono);
        }
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();

        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE,5);
        LSLogisticBoost bestModel = null;
        for (int i = 1; i<=maxIter; i++){
            optimizer.iterate();

            if (i%10==0){
                double mse = MSE.mse(lsLogisticBoost, validation);
                //todo
//                double trainMse = MSE.mse(lsLogisticBoost, regDataSet);
//                System.out.println("iter="+i+", train mse="+trainMse+" , valid mse="+mse);
                earlyStopper.add(i,mse);
                if (earlyStopper.getBestIteration()==i){
                    try {
                        bestModel = (LSLogisticBoost) Serialization.deepCopy(lsLogisticBoost);
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                if (earlyStopper.shouldStop()){
                    break;
                }
            }
        }
        return new Reranker(bestModel, numCandidates,predictionFeatureExtractor);
    }


//
//
//    public Reranker trainLambdaMART(PredictionVectorizer.TrainData trainData, CBM cbm, PredictionVectorizer predictionVectorizer, int ndcgTruncationLevel){
//        LambdaMART lambdaMART = new LambdaMART();
//
//        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf);
//        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
//        LambdaMARTOptimizer optimizer = new LambdaMARTOptimizer(lambdaMART, trainData.regDataSet, trainData.regDataSet.getLabels(), regTreeFactory, trainData.instancesForEachQuery);
//
//        optimizer.setNdcgTruncationLevel(ndcgTruncationLevel);
//
//        if (monotonic){
//            optimizer.setMonotonicity(predictionVectorizer.getMonotonicityConstraints(cbm.getNumClasses()));
//        }
//        optimizer.setShrinkage(shrinkage);
//        optimizer.initialize();
//
//        for (int i=1;i<=numIterations;i++){
//            optimizer.iterate();
//        }
//
//        return new Reranker(lambdaMART, cbm, numCandidates,predictionVectorizer);
//    }


    private RerankerTrainer(Builder builder) {
        numLeaves = builder.numLeaves;
        numCandidates = builder.numCandidates;
        shrinkage = builder.shrinkage;
        minDataPerLeaf = builder.minDataPerLeaf;
        maxIter = builder.maxIter;
        monotonicityType = builder.monotonicityType;

    }

    public static Builder newBuilder() {
        return new Builder();
    }


    public static final class Builder {
        private int numLeaves = 10;
        private int numCandidates = 50;
        private double shrinkage = 0.1;
        private int minDataPerLeaf=5;
        private int maxIter=1000;
        //"none", "weak", "strong", "xgboost"
        private String monotonicityType="none";

        private Builder() {
        }

        public Builder numLeaves(int val) {
            numLeaves = val;
            return this;
        }



        public Builder monotonicityType(String val) {
            monotonicityType = val;
            return this;
        }

        public Builder numCandidates(int val){
            numCandidates = val;
            return this;
        }

        public Builder shrinkage(double val){
            shrinkage = val;
            return this;
        }

        public Builder minDataPerLeaf(int val){
            minDataPerLeaf = val;
            return this;
        }

        public Builder maxIter(int val){
            maxIter = val;
            return this;
        }

        public RerankerTrainer build() {
            return new RerankerTrainer(this);
        }
    }
}
