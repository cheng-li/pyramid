package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 11/16/16.
 */
public class CBMSOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private CBMS cbms;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;

    // format [#data][#components]
    double[][] gammas;

    private boolean isParallel = true;

    // for deterministic annealing
    private double temperature = 1;



    // boosting parameters
    private int numLeavesBinary = 2;
    private int numLeavesMultiClass = 2;
    private double shrinkageBinary = 0.1;
    private double shrinkageMultiClass = 0.1;
    private int numIterationsBinary = 20;
    private int numIterationsMultiClass = 20;

    private DataSet augmentedData;
    private int numComponents;



    public CBMSOptimizer(CBMS cbms, MultiLabelClfDataSet dataSet) {
        this.cbms = cbms;
        this.numComponents = cbms.numComponents;
        this.dataSet = dataSet;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][cbms.getNumComponents()];


        this.augmentedData = DataSetBuilder.getBuilder().numDataPoints(dataSet.getNumDataPoints()*numComponents)
                .numFeatures(dataSet.getNumFeatures()+numComponents)
                .dense(dataSet.isDense())
                .build();
        int dataIndex = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            Vector oldRow = dataSet.getRow(i);
            for (int k=0;k<numComponents;k++){
                for (int j=0;j<dataSet.getNumFeatures();j++){
                    augmentedData.setFeatureValue(dataIndex, j, oldRow.get(j));
                }

                augmentedData.setFeatureValue(dataIndex, dataSet.getNumFeatures()+k, 1);

                dataIndex += 1;
            }
        }


    }



    public void setNumLeavesBinary(int numLeavesBinary) {
        this.numLeavesBinary = numLeavesBinary;
    }

    public void setNumLeavesMultiClass(int numLeavesMultiClass) {
        this.numLeavesMultiClass = numLeavesMultiClass;
    }

    public void setShrinkageBinary(double shrinkageBinary) {
        this.shrinkageBinary = shrinkageBinary;
    }

    public void setShrinkageMultiClass(double shrinkageMultiClass) {
        this.shrinkageMultiClass = shrinkageMultiClass;
    }

    public void setNumIterationsBinary(int numIterationsBinary) {
        this.numIterationsBinary = numIterationsBinary;
    }

    public void setNumIterationsMultiClass(int numIterationsMultiClass) {
        this.numIterationsMultiClass = numIterationsMultiClass;
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public void optimize() {
        while (true) {
            iterate();
            if (terminator.shouldTerminate()) {
                break;
            }
        }
    }

    public void iterate() {
        eStep();
        System.out.println("gamma = "+ Arrays.toString(gammas[0]));
        mStep();
        this.terminator.add(getObjective());
    }

    private void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        updateGamma();

        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+getObjective());
        }
    }


    private void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    private void updateGamma(int n) {
        Vector x = dataSet.getRow(n);
        MultiLabel y = dataSet.getMultiLabels()[n];
        double[] posterior = cbms.posteriorMembership(x, y);
        for (int k = 0; k< cbms.numComponents; k++) {
            gammas[n][k] = posterior[k];
        }
    }

    void mStep() {
        if (logger.isDebugEnabled()){
            logger.debug("start M step");
        }
        updateBinaryClassifiers();
        updateMultiClassClassifier();
        if (logger.isDebugEnabled()){
            logger.debug("finish M step");
            logger.debug("objective = "+getObjective());
        }
    }

    private void updateBinaryClassifiers() {
        if (logger.isDebugEnabled()){
            logger.debug("start updateBinaryClassifiers");
        }
        IntStream.range(0, cbms.numLabels).forEach(l -> updateBinaryBoosting(l));
        if (logger.isDebugEnabled()){
            logger.debug("finish updateBinaryClassifiers");
        }
    }

    //todo pay attention to parallelism


    private void updateBinaryBoosting(int labelIndex){
//        System.out.println("updating binary boosting for class + "+labelIndex);
        double[][] targets = new double[augmentedData.getNumDataPoints()][2];
        int[] binaryLabels = new int[augmentedData.getNumDataPoints()];

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            boolean match = multiLabel.matchClass(labelIndex);
            for (int k=0;k<numComponents;k++){
                if (match){
                    targets[i*numComponents+k][1]=1;
                    binaryLabels[i*numComponents+k]=1;
                } else {
                    targets[i*numComponents+k][0]=1;
                }
            }
        }

        double[] weights = new double[augmentedData.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<numComponents;k++){
                weights[i*numComponents+k] = gammas[i][k];
            }
        }
//        System.out.println("for label "+labelIndex);
//        System.out.println("targets = "+Arrays.deepToString(targets));
//        System.out.println("weights = "+Arrays.toString(weights));



        int numIterations = numIterationsBinary;
        double shrinkage = shrinkageBinary;
        LKBoost boost = (LKBoost)this.cbms.binaryClassifiers[labelIndex];
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeavesBinary);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(2));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost,augmentedData, regTreeFactory,
                weights,targets);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        optimizer.iterate(numIterations);
//        System.out.println("training accu = "+ Accuracy.accuracy(binaryLabels, boost.predict(augmentedData)));


    }



    private void updateMultiClassClassifier(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateMultiClassClassifier()");
        }
        updateMultiClassBoost();
    }


    private void updateMultiClassBoost() {
        int numComponents = cbms.numComponents;
        int numIterations = numIterationsMultiClass;
        double shrinkage = shrinkageMultiClass;
        LKBoost boost = (LKBoost)this.cbms.multiClassClassifier;
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeavesMultiClass);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numComponents));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost, dataSet, regTreeFactory, gammas);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        optimizer.iterate(numIterations);
    }




    //TODO: have to modify the objectives by introducing L1 regularization part
    public double getObjective() {
        return multiClassClassifierObj() + binaryObj() +(1-temperature)*getEntropy();
    }


    private double getEntropy() {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i) {
        return Entropy.entropy(gammas[i]);
    }


    private double binaryObj(){
        return IntStream.range(0, cbms.numLabels).mapToDouble(this::binaryObj).sum();
    }



    private double binaryObj(int classIndex){
        return binaryBoostObj(classIndex);
    }



    private double binaryBoostObj(int classIndex){
        double[][] targets = new double[augmentedData.getNumDataPoints()][2];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            boolean match = multiLabel.matchClass(classIndex);
            for (int k=0;k<numComponents;k++){
                if (match){
                    targets[i*numComponents+k][1]=1;
                } else {
                    targets[i*numComponents+k][0]=1;
                }
            }
        }

        double[] weights = new double[augmentedData.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<numComponents;k++){
                weights[i*numComponents+k] = gammas[i][k];
            }
        }

        Classifier.ProbabilityEstimator estimator = cbms.binaryClassifiers[classIndex];

        return KLDivergence.kl(estimator, augmentedData, targets, weights);
    }

    private double multiClassClassifierObj(){
        return multiClassBoostObj();
    }



    private double multiClassBoostObj(){
        Classifier.ProbabilityEstimator estimator = cbms.multiClassClassifier;
        double[][] targets = gammas;
        return KLDivergence.kl(estimator,dataSet,targets);
    }



    public Terminator getTerminator() {
        return terminator;
    }

    public double[][] getGammas() {
        return gammas;
    }

    public double[][] getPIs() {
        double[][] PIs = new double[dataSet.getNumDataPoints()][cbms.getNumComponents()];

        for (int n=0; n<PIs.length; n++) {
            double[] logProbs = cbms.multiClassClassifier.predictLogClassProbs(dataSet.getRow(n));
            for (int k=0; k<PIs[n].length; k++) {
                PIs[n][k] = Math.exp(logProbs[k]);
            }
        }
        return PIs;
    }
}
