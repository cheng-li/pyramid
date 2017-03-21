package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.LBFGS;
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


//    // for deterministic annealing
//    private double temperature = 1;


    // lr parameters
    // regularization for multiClassClassifier
    private double priorVarianceMultiClass =1;
    // regularization for binary logisticRegression
    private double priorVarianceBinary =1;

    private double componentWeightsVariance=1;

    // in each M step
    private int numMultiClassParaUpdates = 10;
    private int numBinaryParaUpdates = 10;

    public CBMSOptimizer(CBMS cbms, MultiLabelClfDataSet dataSet) {
        this.cbms = cbms;
        this.dataSet = dataSet;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][cbms.getNumComponents()];
    }


    public void setNumMultiClassParaUpdates(int numMultiClassParaUpdates) {
        this.numMultiClassParaUpdates = numMultiClassParaUpdates;
    }

    public void setNumBinaryParaUpdates(int numBinaryParaUpdates) {
        this.numBinaryParaUpdates = numBinaryParaUpdates;
    }

    public void setComponentWeightsVariance(double componentWeightsVariance) {
        this.componentWeightsVariance = componentWeightsVariance;
    }

    public void setPriorVarianceMultiClass(double priorVarianceMultiClass) {
        this.priorVarianceMultiClass = priorVarianceMultiClass;
    }

    public void setPriorVarianceBinary(double priorVarianceBinary) {
        this.priorVarianceBinary = priorVarianceBinary;
    }



//    public double getTemperature() {
//        return temperature;
//    }
//
//    public void setTemperature(double temperature) {
//        this.temperature = temperature;
//    }

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
        // do not compute objective by default
//        this.terminator.add(getObjective());
    }

    public void eStep(){
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

    public void mStep() {
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
        IntStream.range(0, cbms.numLabels)
                .parallel()
                .forEach(this::updateBinaryLogisticRegression);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateBinaryClassifiers");
        }
    }

    //todo pay attention to parallelism


    private void updateBinaryLogisticRegression(int labelIndex){
        AugmentedLRLoss loss = new AugmentedLRLoss(dataSet, labelIndex, gammas,
                cbms.getBinaryClassifiers()[labelIndex],priorVarianceBinary, componentWeightsVariance);
        LBFGS lbfgs = new LBFGS(loss);
        //todo
        lbfgs.getTerminator().setMaxIteration(numBinaryParaUpdates);
        lbfgs.getTerminator().setGoal(Terminator.Goal.MINIMIZE);
        lbfgs.optimize();
    }

    private void updateMultiClassLR() {
        // parallel
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbms.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        //TODO maximum iterations
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(numMultiClassParaUpdates);
        ridgeLogisticOptimizer.optimize();
    }

    private void updateMultiClassClassifier(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateMultiClassClassifier()");
        }
        updateMultiClassLR();
    }



    //TODO: have to modify the objectives by introducing L1 regularization part
    public double getObjective() {
        return multiClassClassifierObj() + binaryObj();
//                +(1-temperature)*getEntropy();
    }


    private double getEntropy() {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i) {
        return Entropy.entropy(gammas[i]);
    }


    double binaryObj(){
        return IntStream.range(0, cbms.numLabels).parallel().mapToDouble(this::binaryObj).sum();
    }


    private double binaryObj(int labelIndex){
        AugmentedLRLoss loss = new AugmentedLRLoss(dataSet, labelIndex, gammas,
                cbms.getBinaryClassifiers()[labelIndex],priorVarianceBinary, componentWeightsVariance);
        return loss.getValue();
    }


    double multiClassClassifierObj(){
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbms.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        return ridgeLogisticOptimizer.getFunction().getValue();
    }


    public Terminator getTerminator() {
        return terminator;
    }

    public double[][] getGammas() {
        return gammas;
    }

}
