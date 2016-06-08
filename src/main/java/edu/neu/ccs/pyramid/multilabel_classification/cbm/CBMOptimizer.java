package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.*;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * References
 * Deterministic annealing:
 * Katahira, Kentaro, Kazuho Watanabe, and Masato Okada.
 * "Deterministic annealing variant of variational Bayes method."
 * Journal of Physics: Conference Series. Vol. 95. No. 1. IOP Publishing, 2008.
 *
 * Created by Rainicy on 10/23/15.
 */
public class CBMOptimizer implements Serializable, Parallelizable {
    private static final Logger logger = LogManager.getLogger();
    private CBM CBM;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;

    // format [#data][#cluster]
    double[][] gammas;
    // format [#cluster][#data]
    double[][] gammasT;

    // format [#data]
    private Vector[] labels;

    // format [#labels][#data][2]
    private double[][][] targetsDistributions;
    private boolean isParallel = true;

    // for deterministic annealing
    private double temperature = 1;


    // if hard assignment or soft assignment
    private boolean hardAssignment = false;

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

    private double meanRegVariance = 10000;

    private boolean meanRegularization = false;

    // boosting parameters
    private int numLeavesBinary = 2;
    private int numLeavesMultiClass = 2;
    private double shrinkageBinary = 0.1;
    private double shrinkageMultiClass = 0.1;
    private int numIterationsBinary = 20;
    private int numIterationsMultiClass = 20;



    public CBMOptimizer(CBM CBM, MultiLabelClfDataSet dataSet) {
        this.CBM = CBM;
        this.dataSet = dataSet;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][CBM.numClusters];

        this.gammasT = new double[CBM.numClusters][dataSet.getNumDataPoints()];
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            for (int k = 0; k< CBM.numClusters; k++){
                gammas[n][k] = 1.0/ CBM.numClusters;
                gammasT[k][n] = 1.0/ CBM.numClusters;
            }
        }
        this.labels = new DenseVector[dataSet.getNumDataPoints()];
        for (int n=0; n<labels.length; n++) {
            Set<Integer> label = dataSet.getMultiLabels()[n].getMatchedLabels();
            labels[n] = new DenseVector(dataSet.getNumClasses());
            for (int l : label) {
                labels[n].set(l, 1);
            }
        }

        // 2 means binary classification problem.
        this.targetsDistributions = new double[CBM.getNumClasses()][dataSet.getNumDataPoints()][2];
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            Vector label = labels[n];
            for (int l=0; l<label.size(); l++) {
                if (label.get(l) == 0.0) {
                    this.targetsDistributions[l][n][0] = 1;
                } else {
                    this.targetsDistributions[l][n][1] = 1;
                }
            }
        }
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

    public void setMeanRegVariance(double meanRegVariance) {
        this.meanRegVariance = meanRegVariance;
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public void setHardAssignment(boolean hardAssignment) {
        this.hardAssignment = hardAssignment;
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

    public void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        updateGamma();
        // reweightedGammas
        if (hardAssignment) {
            reweightedGammas();
        }
        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+getObjective());
        }
    }

    @Override
    public void setParallelism(boolean isParallel) {
        this.isParallel = isParallel;
    }

    public void setMeanRegularization(boolean meanRegularization) {
        this.meanRegularization = meanRegularization;
    }

    @Override
    public boolean isParallel() {
        return this.isParallel;
    }

    private void reweightedGammas() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::reweightedGammas);
    }

    private void reweightedGammas(int n) {

        // find the max gamma index
        int maxK = 0;
        double maxGam = gammas[n][0];
        for (int k=1; k<gammas[n].length; k++) {
            if (maxGam < gammas[n][k]) {
                maxGam = gammas[n][k];
                maxK = k;
            }
        }

        // reweighted
        for (int k=0; k<gammas[n].length; k++) {
            if (k == maxK) {
                gammas[n][k] = 1.0;
                gammasT[k][n] = 1.0;
            } else {
                gammas[n][k] = 0.0;
                gammasT[k][n] = 0.0;
            }
        }

//        int[] indexes = maxKIndex(gammas[n],3);
//        Set<Integer> indexSet = new HashSet<>();
//        for (int topK : indexes) {
//            indexSet.add(topK);
//        }
//
//        //reweighted
//        double restValues = 0.0;
//        for (int k=0; k<gammas[n].length; k++) {
//            if (!indexSet.contains(k)) {
//                restValues += gammas[n][k];
//                gammas[n][k] = 0.0;
//                gammasT[k][n] = 0.0;
//            }
//        }
//        double averageValue = restValues / 3.0;
//        for (int topK : indexSet) {
//            gammas[n][topK] += averageValue;
//            gammasT[topK][n] += averageValue;
//        }
    }

//    private static int[] maxKIndex(double[] array, int top_k) {
//        double[] max = new double[top_k];
//        int[] maxIndex = new int[top_k];
//        Arrays.fill(max, Double.NEGATIVE_INFINITY);
//        Arrays.fill(maxIndex, -1);
//
//        top: for(int i = 0; i < array.length; i++) {
//            for(int j = 0; j < top_k; j++) {
//                if(array[i] > max[j]) {
//                    for(int x = top_k - 1; x > j; x--) {
//                        maxIndex[x] = maxIndex[x-1]; max[x] = max[x-1];
//                    }
//                    maxIndex[j] = i; max[j] = array[i];
//                    continue top;
//                }
//            }
//        }
//        return maxIndex;
//    }

    private void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    private void updateGamma(int n) {
        Vector X = dataSet.getRow(n);
        Vector Y = this.labels[n];
        int K = CBM.numClusters;
        // log[p(z_n=k | x_n)] array
        double[] logLogisticProbs = CBM.multiClassClassifier.predictLogClassProbs(X);
        // log[p(y_n | z_n=k, x_n)] for all k from 1 to K;
        double[] logClusterConditionalProbs = CBM.clusterConditionalLogProbArr(X, Y);
        double[] logNumerators = new double[logLogisticProbs.length];
        for (int k=0; k<K; k++) {
            logNumerators[k] = (logLogisticProbs[k] + logClusterConditionalProbs[k])/temperature;
        }
        double logDenominator = MathUtil.logSumExp(logNumerators);
        for (int k=0; k<K; k++) {
            double value = Math.exp(logNumerators[k] - logDenominator);
            gammas[n][k] = value;
            gammasT[k][n] = value;
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
        IntStream.range(0, CBM.numClusters).forEach(this::updateBinaryClassifiers);

    }

    //todo pay attention to parallelism
    private void updateBinaryClassifiers(int clusterIndex){
        String type = CBM.getBinaryClassifierType();
        switch (type){
            case "lr":
                IntStream.range(0, CBM.numLabels).parallel().forEach(l-> updateBinaryLogisticRegression(clusterIndex,l));
                break;
            case "boost":
                // no parallel for boosting
                IntStream.range(0, CBM.numLabels).forEach(l -> updateBinaryBoosting(clusterIndex, l));
                break;
            case "elasticnet":
                IntStream.range(0, CBM.numLabels).parallel().forEach(l-> updateBinaryLogisticRegressionEL(clusterIndex,l));
                break;
            default:
                throw new IllegalArgumentException("unknown type: " + CBM.getBinaryClassifierType());
        }
    }

    private void updateBinaryBoosting(int clusterIndex, int labelIndex){
        int numIterations = numIterationsBinary;
        double shrinkage = shrinkageBinary;
        LKBoost boost = (LKBoost)this.CBM.binaryClassifiers[clusterIndex][labelIndex];
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeavesBinary);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(2));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost,dataSet, regTreeFactory,gammasT[clusterIndex],targetsDistributions[labelIndex]);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        for (int i=0;i<numIterations;i++){
            optimizer.iterate();
        }
    }

    private void updateBinaryLogisticRegression(int clusterIndex, int labelIndex){
        RidgeLogisticOptimizer ridgeLogisticOptimizer;
        if (meanRegularization){
            Weights mean = CBMInspector.getMean(CBM,labelIndex);
            Weights zero = new Weights(2,dataSet.getNumFeatures());
            List<Weights> means = new ArrayList<>();
            means.add(zero);
            means.add(mean);
            List<Double> variances = new ArrayList<>();
            //todo two
            variances.add(priorVarianceBinary);
            variances.add(meanRegVariance);
            LogisticLoss logisticLoss = new LogisticLoss((LogisticRegression) CBM.binaryClassifiers[clusterIndex][labelIndex],
                    dataSet, gammasT[clusterIndex], targetsDistributions[labelIndex], means,variances);
            ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticLoss);
        } else {
            ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression) CBM.binaryClassifiers[clusterIndex][labelIndex],
                    dataSet, gammasT[clusterIndex], targetsDistributions[labelIndex], priorVarianceBinary);
        }
        //TODO
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(10);
        ridgeLogisticOptimizer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("for cluster "+clusterIndex+" label "+labelIndex+" history= "+ridgeLogisticOptimizer.getOptimizer().getTerminator().getHistory());
        }
    }

    private void updateBinaryLogisticRegressionEL(int clusterIndex, int labelIndex) {
        ElasticNetLogisticTrainer elasticNetLogisticTrainer = new ElasticNetLogisticTrainer.Builder((LogisticRegression)
                CBM.binaryClassifiers[clusterIndex][labelIndex], dataSet, 2, targetsDistributions[labelIndex])
                .setRegularization(regularizationBinary)
                .setL1Ratio(l1RatioBinary)
                .setLineSearch(lineSearch).build();
        //TODO: maximum iterations.
        elasticNetLogisticTrainer.getTerminator().setMaxIteration(10);
        elasticNetLogisticTrainer.optimize();
    }

    private void updateMultiClassClassifier(){
        String type = CBM.getMultiClassClassifierType();
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
                throw new IllegalArgumentException("unknown type: " + CBM.getMultiClassClassifierType());
        }
    }

    private void updateMultiClassEL() {
        ElasticNetLogisticTrainer elasticNetLogisticTrainer = new ElasticNetLogisticTrainer.Builder((LogisticRegression)
        CBM.multiClassClassifier, dataSet, CBM.multiClassClassifier.getNumClasses(), gammas)
                .setRegularization(regularizationMultiClass)
                .setL1Ratio(l1RatioMultiClass)
                .setLineSearch(lineSearch).build();
        // TODO: maximum iterations
        elasticNetLogisticTrainer.getTerminator().setMaxIteration(10);
        elasticNetLogisticTrainer.optimize();
    }

    private void updateMultiClassLR() {
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression) CBM.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass);
        ridgeLogisticOptimizer.setParallelism(true);
        //TODO
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(10);
        ridgeLogisticOptimizer.optimize();
    }

    private void updateMultiClassBoost() {
        int numClusters = CBM.numClusters;
        int numIterations = numIterationsMultiClass;
        double shrinkage = shrinkageMultiClass;
        LKBoost boost = (LKBoost)this.CBM.multiClassClassifier;
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeavesMultiClass);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numClusters));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost,dataSet, regTreeFactory,gammas);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        for (int i=0;i<numIterations;i++){
            optimizer.iterate();
        }
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
//
    private double getEntropy() {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i) {
        return Entropy.entropy(gammas[i]);
    }


    private double binaryObj(){
        return IntStream.range(0, CBM.numClusters).mapToDouble(this::binaryObj).sum();
    }

    private double binaryObj(int clusterIndex){
        return IntStream.range(0, CBM.numLabels).parallel().mapToDouble(l->binaryObj(clusterIndex,l)).sum();
    }

    private double binaryObj(int clusterIndex, int classIndex){
        String type = CBM.getBinaryClassifierType();
        switch (type){
            case "lr":
                return binaryLRObj(clusterIndex, classIndex);
            case "boost":
                return binaryBoostObj(clusterIndex, classIndex);
            case "elasticnet":
                return binaryLRObj(clusterIndex, classIndex);
            default:
                throw new IllegalArgumentException("unknown type: " + type);
        }
    }

    //todo mean regularization is not handled here
    // consider regularization penalty
    private double binaryLRObj(int clusterIndex, int classIndex) {
            LogisticLoss logisticLoss = new LogisticLoss((LogisticRegression) CBM.binaryClassifiers[clusterIndex][classIndex],
                    dataSet, gammasT[clusterIndex], targetsDistributions[classIndex], priorVarianceBinary);
            return logisticLoss.getValue();
    }

    private double binaryBoostObj(int clusterIndex, int classIndex){
        Classifier.ProbabilityEstimator estimator = CBM.binaryClassifiers[clusterIndex][classIndex];
        double[][] targets = targetsDistributions[classIndex];
        double[] weights = gammasT[clusterIndex];
        return KLDivergence.kl(estimator, dataSet, targets, weights);
    }

    private double multiClassClassifierObj(){
        String type = CBM.getMultiClassClassifierType();
        switch (type){
            case "lr":
                return multiClassLRObj();
            case "boost":
                return multiClassBoostObj();
            //TODO: change to elastic net
            case "elasticnet":
                return multiClassLRObj();
            default:
                throw new IllegalArgumentException("unknown type: " + type);
        }
    }

    private double multiClassBoostObj(){
        Classifier.ProbabilityEstimator estimator = CBM.multiClassClassifier;
        double[][] targets = gammas;
        return KLDivergence.kl(estimator,dataSet,targets);
    }

    private double multiClassLRObj(){
        LogisticLoss logisticLoss =  new LogisticLoss((LogisticRegression) CBM.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass);
        return logisticLoss.getValue();
    }


    public Terminator getTerminator() {
        return terminator;
    }

    public double[][] getGammas() {
        return gammas;
    }

    public double[][] getPIs() {
        double[][] PIs = new double[dataSet.getNumDataPoints()][CBM.getNumClusters()];

        for (int n=0; n<PIs.length; n++) {
            double[] logProbs = CBM.multiClassClassifier.predictLogClassProbs(dataSet.getRow(n));
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
}
