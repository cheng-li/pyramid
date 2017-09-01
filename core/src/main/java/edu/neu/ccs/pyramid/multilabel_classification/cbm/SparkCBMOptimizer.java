package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 11/28/16.
 */
public class SparkCBMOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private CBM cbm;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;

    // format [#data][#components]
    double[][] gammas;
    // format [#components][#data]
    double[][] gammasT;

    // format [#labels][#data][2]
    // to be fit by binary classifiers
    private double[][][] targetsDistributions;


    // for deterministic annealing
    private double temperature = 1;


    // lr parameters
    // regularization for multiClassClassifier
    private double priorVarianceMultiClass =1;
    // regularization for binary logisticRegression
    private double priorVarianceBinary =1;

    // elasticnet parameters
    private double regularizationMultiClass = 1.0;
    private double regularizationBinary = 1.0;
    private double l1RatioBinary = 0.0;
    private double l1RatioMultiClass = 0.0;
    private boolean lineSearch = true;


    // boosting parameters
    private int numLeavesBinary = 2;
    private int numLeavesMultiClass = 2;
    private double shrinkageBinary = 0.1;
    private double shrinkageMultiClass = 0.1;
    private int numIterationsBinary = 20;
    private int numIterationsMultiClass = 20;

    //spark
    private JavaSparkContext sparkContext;
    private Broadcast<MultiLabelClfDataSet> dataSetBroadCast;
    private Broadcast<double[][][]> targetDisBroadCast;



    public SparkCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet, JavaSparkContext sparkContext) {
        this.cbm = cbm;
        this.dataSet = dataSet;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][cbm.getNumComponents()];
        this.gammasT = new double[cbm.getNumComponents()][dataSet.getNumDataPoints()];
        double average = 1.0/ cbm.getNumComponents();
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            for (int k = 0; k< cbm.getNumComponents(); k++){
                gammas[n][k] = average;
                gammasT[k][n] = average;
            }
        }


        this.targetsDistributions = new double[cbm.getNumClasses()][dataSet.getNumDataPoints()][2];
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            // first mark all labels as negative
            for (int l=0;l<cbm.getNumClasses(); l++){
                this.targetsDistributions[l][n][0] = 1;
            }
            MultiLabel multiLabel = dataSet.getMultiLabels()[n];
            for (int l: multiLabel.getMatchedLabels()){
                this.targetsDistributions[l][n][0] = 0;
                this.targetsDistributions[l][n][1] = 1;
            }
        }

        this.sparkContext = sparkContext;
        this.dataSetBroadCast = sparkContext.broadcast(dataSet);
        this.targetDisBroadCast = sparkContext.broadcast(targetsDistributions);
    }

    public void setPriorVarianceMultiClass(double priorVarianceMultiClass) {
        this.priorVarianceMultiClass = priorVarianceMultiClass;
    }

    public void setPriorVarianceBinary(double priorVarianceBinary) {
        this.priorVarianceBinary = priorVarianceBinary;
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
        double[] posterior = cbm.posteriorMembership(x, y);
        for (int k=0; k<cbm.numComponents; k++) {
            gammas[n][k] = posterior[k];
            gammasT[k][n] = posterior[k];
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



        Classifier.ProbabilityEstimator[][] localBinaryClassifiers = cbm.binaryClassifiers;
        double[][] localGammasT  = gammasT;
        Broadcast<MultiLabelClfDataSet> localDataSetBroadcast = dataSetBroadCast;
        Broadcast<double[][][]> localTargetsBroadcast = targetDisBroadCast;
        double localVariance = priorVarianceBinary;


        List<BinaryTask> binaryTaskList = new ArrayList<>();
        for (int k=0;k<cbm.numComponents;k++){
            for (int l=0;l<cbm.numLabels;l++){
                LogisticRegression logisticRegression = (LogisticRegression)localBinaryClassifiers[k][l];
                double[] weights = localGammasT[k];
                binaryTaskList.add(new BinaryTask(k,l, logisticRegression, weights));
            }
        }

        JavaRDD<BinaryTask> binaryTaskRDD = sparkContext.parallelize(binaryTaskList, binaryTaskList.size());
        List<BinaryTaskResult> results = binaryTaskRDD.map(binaryTask-> {
            int labelIndex = binaryTask.classIndex;
            //todo move this to rdd
            // each element in rdd should contain its full information
            return updateBinaryLogisticRegression(binaryTask.componentIndex, binaryTask.classIndex, binaryTask.logisticRegression,
                    localDataSetBroadcast.value(), binaryTask.weights, localTargetsBroadcast.value()[labelIndex],localVariance);

        })
        .collect();
        for (BinaryTaskResult result: results){
            cbm.binaryClassifiers[result.componentIndex][result.classIndex] = result.binaryClassifier;
        }


//        IntStream.range(0, cbm.numComponents).forEach(this::updateBinaryClassifiers);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateBinaryClassifiers");
        }
    }

//    //todo pay attention to parallelism
//    private void updateBinaryClassifiers(int component){
//        String type = cbm.getBinaryClassifierType();
//        switch (type){
//            case "lr":
//                IntStream.range(0, cbm.numLabels).parallel().forEach(l-> updateBinaryLogisticRegression(component,l));
//                break;
//            case "boost":
//                // no parallel for boosting
//                IntStream.range(0, cbm.numLabels).forEach(l -> updateBinaryBoosting(component, l));
//                break;
//            case "elasticnet":
//                IntStream.range(0, cbm.numLabels).parallel().forEach(l-> updateBinaryLogisticRegressionEL(component,l));
//                break;
//            default:
//                throw new IllegalArgumentException("unknown type: " + cbm.getBinaryClassifierType());
//        }
//    }

//    private void updateBinaryBoosting(int componentIndex, int labelIndex){
//        int numIterations = numIterationsBinary;
//        double shrinkage = shrinkageBinary;
//        LKBoost boost = (LKBoost)this.cbm.binaryClassifiers[componentIndex][labelIndex];
//        RegTreeConfig regTreeConfig = new RegTreeConfig()
//                .setMaxNumLeaves(numLeavesBinary);
//        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
//        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(2));
//        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost,dataSet, regTreeFactory,
//                gammasT[componentIndex],targetsDistributions[labelIndex]);
//        optimizer.setShrinkage(shrinkage);
//        optimizer.initialize();
//        optimizer.iterate(numIterations);
//    }




    private static BinaryTaskResult updateBinaryLogisticRegression(int componentIndex, int labelIndex, LogisticRegression logisticRegression,
                                                            MultiLabelClfDataSet dataSet, double[] weights,
                                                            double[][] targets, double variance){
        RidgeLogisticOptimizer ridgeLogisticOptimizer;
        // no parallelism
        ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticRegression,
                dataSet, weights,
                targets, variance, false);
        //TODO maximum iterations
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(15);
        ridgeLogisticOptimizer.optimize();
        return new BinaryTaskResult(componentIndex, labelIndex, logisticRegression);
//        if (logger.isDebugEnabled()){
//            logger.debug("for cluster "+clusterIndex+" label "+labelIndex+" history= "+ridgeLogisticOptimizer.getOptimizer().getTerminator().getHistory());
//        }
    }

//    private void updateBinaryLogisticRegressionEL(int componentIndex, int labelIndex) {
//        ElasticNetLogisticTrainer elasticNetLogisticTrainer = new ElasticNetLogisticTrainer.Builder((LogisticRegression)
//                cbm.binaryClassifiers[componentIndex][labelIndex], dataSet, 2, targetsDistributions[labelIndex], gammasT[componentIndex])
//                .setRegularization(regularizationBinary)
//                .setL1Ratio(l1RatioBinary)
//                .setLineSearch(lineSearch).build();
//        //TODO: maximum iterations
//        elasticNetLogisticTrainer.getTerminator().setMaxIteration(15);
//        elasticNetLogisticTrainer.optimize();
//    }

    private void updateMultiClassClassifier(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateMultiClassClassifier()");
        }
        String type = cbm.getMultiClassClassifierType();
        switch (type){
            case "lr":
                updateMultiClassLR();
                break;
            case "boost":
                updateMultiClassBoost();
                break;
            case "elasticnet":
                updateMultiClassEL();
                break;
            default:
                throw new IllegalArgumentException("unknown type: " + cbm.getMultiClassClassifierType());
        }
        if (logger.isDebugEnabled()){
            logger.debug("finish updateMultiClassClassifier()");
        }
    }

    private void updateMultiClassEL() {
        ElasticNetLogisticTrainer elasticNetLogisticTrainer = new ElasticNetLogisticTrainer.Builder((LogisticRegression)
                cbm.multiClassClassifier, dataSet, cbm.multiClassClassifier.getNumClasses(), gammas)
                .setRegularization(regularizationMultiClass)
                .setL1Ratio(l1RatioMultiClass)
                .setLineSearch(lineSearch).build();
        // TODO: maximum iterations
        elasticNetLogisticTrainer.getTerminator().setMaxIteration(15);
        elasticNetLogisticTrainer.optimize();
    }

    private void updateMultiClassLR() {
        // parallel
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbm.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        //TODO maximum iterations
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(15);
        ridgeLogisticOptimizer.optimize();
    }

    private void updateMultiClassBoost() {
        int numComponents = cbm.numComponents;
        int numIterations = numIterationsMultiClass;
        double shrinkage = shrinkageMultiClass;
        LKBoost boost = (LKBoost)this.cbm.multiClassClassifier;
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeavesMultiClass);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numComponents));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost, dataSet, regTreeFactory, gammas);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        optimizer.iterate(numIterations);
    }

//    public static Object[] getColumn(Object[][] array, int index){
//        Object[] column = new Object[array[0].length]; // Here I assume a rectangular 2D array!
//        for(int i=0; i<column.length; i++){
//            column[i] = array[i][index];
//        }
//        return column;
//    }


    //TODO: have to modify the objectives by introducing L1 regularization part
    public double getObjective() {
        return multiClassClassifierObj() + binaryObj() +(1-temperature)*getEntropy();
    }

//    private double getMStepObjective() {
//        KLLogisticLoss logisticLoss =  new KLLogisticLoss(bmmClassifier.multiClassClassifier,
//                dataSet, gammas, priorVarianceMultiClass);
//        // Q function for \Thata + gamma.entropy and Q function for Weights
//        return logisticLoss.getValue() + binaryLRObj();
//    }

    private double getEntropy() {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i) {
        return Entropy.entropy(gammas[i]);
    }


    private double binaryObj(){
        return IntStream.range(0, cbm.numComponents).mapToDouble(this::binaryObj).sum();
    }

    private double binaryObj(int clusterIndex){
        return IntStream.range(0, cbm.numLabels).parallel().mapToDouble(l->binaryObj(clusterIndex,l)).sum();
    }

    private double binaryObj(int clusterIndex, int classIndex){
        String type = cbm.getBinaryClassifierType();
        switch (type){
            case "lr":
                return binaryLRObj(clusterIndex, classIndex);
            case "boost":
                return binaryBoostObj(clusterIndex, classIndex);
            case "elasticnet":
                // todo
                return binaryLRELObj(clusterIndex, classIndex);
            default:
                throw new IllegalArgumentException("unknown type: " + type);
        }
    }

    private double binaryLRELObj(int clusterIndex, int classIndex) {
        LogisticLoss logisticLoss = new LogisticLoss((LogisticRegression) cbm.binaryClassifiers[clusterIndex][classIndex],
                dataSet, gammasT[clusterIndex], targetsDistributions[classIndex], regularizationBinary, l1RatioBinary, false);
        return logisticLoss.getValueEL();
    }

    // consider regularization penalty
    private double binaryLRObj(int clusterIndex, int classIndex) {
        LogisticLoss logisticLoss = new LogisticLoss((LogisticRegression) cbm.binaryClassifiers[clusterIndex][classIndex],
                dataSet, gammasT[clusterIndex], targetsDistributions[classIndex], priorVarianceBinary, false);
        return logisticLoss.getValue();
    }

    private double binaryBoostObj(int clusterIndex, int classIndex){
        Classifier.ProbabilityEstimator estimator = cbm.binaryClassifiers[clusterIndex][classIndex];
        double[][] targets = targetsDistributions[classIndex];
        double[] weights = gammasT[clusterIndex];
        return KLDivergence.kl(estimator, dataSet, targets, weights);
    }

    private double multiClassClassifierObj(){
        String type = cbm.getMultiClassClassifierType();
        switch (type){
            case "lr":
                return multiClassLRObj();
            case "boost":
                return multiClassBoostObj();
            //TODO: change to elastic net
            case "elasticnet":
                return multiClassLRELObj();
            default:
                throw new IllegalArgumentException("unknown type: " + type);
        }
    }

    private double multiClassLRELObj() {
        LogisticLoss logisticLoss =  new LogisticLoss((LogisticRegression) cbm.multiClassClassifier,
                dataSet, gammas, regularizationMultiClass, l1RatioMultiClass, true);
        return logisticLoss.getValueEL();
    }

    private double multiClassBoostObj(){
        Classifier.ProbabilityEstimator estimator = cbm.multiClassClassifier;
        double[][] targets = gammas;
        return KLDivergence.kl(estimator,dataSet,targets);
    }

    private double multiClassLRObj(){
        LogisticLoss logisticLoss =  new LogisticLoss((LogisticRegression) cbm.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        return logisticLoss.getValue();
    }


    public Terminator getTerminator() {
        return terminator;
    }

    public double[][] getGammas() {
        return gammas;
    }

    public double[][] getPIs() {
        double[][] PIs = new double[dataSet.getNumDataPoints()][cbm.getNumComponents()];

        for (int n=0; n<PIs.length; n++) {
            double[] logProbs = cbm.multiClassClassifier.predictLogClassProbs(dataSet.getRow(n));
            for (int k=0; k<PIs[n].length; k++) {
                PIs[n][k] = Math.exp(logProbs[k]);
            }
        }
        return PIs;
    }

    // For ElasticEet Parameters
    public double getRegularizationMultiClass() {
        return regularizationMultiClass;
    }

    public void setRegularizationMultiClass(double regularizationMultiClass) {
        this.regularizationMultiClass = regularizationMultiClass;
    }

    public double getRegularizationBinary() {
        return regularizationBinary;
    }

    public void setRegularizationBinary(double regularizationBinary) {
        this.regularizationBinary = regularizationBinary;
    }

    public boolean isLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(boolean lineSearch) {
        this.lineSearch = lineSearch;
    }

    public double getL1RatioBinary() {
        return l1RatioBinary;
    }

    public void setL1RatioBinary(double l1RatioBinary) {
        this.l1RatioBinary = l1RatioBinary;
    }

    public double getL1RatioMultiClass() {
        return l1RatioMultiClass;
    }

    public void setL1RatioMultiClass(double l1RatioMultiClass) {
        this.l1RatioMultiClass = l1RatioMultiClass;
    }

    private static class BinaryTask implements Serializable{
        public BinaryTask(int componentIndex, int classIndex, LogisticRegression logisticRegression, double[] weights) {
            this.componentIndex = componentIndex;
            this.classIndex = classIndex;
            this.logisticRegression = logisticRegression;
            this.weights = weights;
        }

        int componentIndex;
        int classIndex;
        LogisticRegression logisticRegression;
        double[] weights;
    }


    private static class BinaryTaskResult implements Serializable{
        public BinaryTaskResult(int componentIndex, int classIndex, Classifier.ProbabilityEstimator binaryClassifier) {
            this.componentIndex = componentIndex;
            this.classIndex = classIndex;
            this.binaryClassifier = binaryClassifier;
        }

        int componentIndex;
        int classIndex;
        Classifier.ProbabilityEstimator binaryClassifier;
    }
}
