package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMOptimizer implements Serializable{
    private static final Logger logger = LogManager.getLogger();
    private transient BMMClassifier bmmClassifier;
    private transient MultiLabelClfDataSet dataSet;
    private transient Terminator terminator;
    // format [data][cluster]
    public double[][] gammas;
    // big variance means small regularization
    private double gaussianPriorVariance;

    private transient Vector[] labels;


    public BMMOptimizer(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet,
                        double gaussianPriorVariance) {
        this.bmmClassifier = bmmClassifier;
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][bmmClassifier.numClusters];

        this.labels = new DenseVector[dataSet.getNumDataPoints()];
        for (int n=0; n<labels.length; n++) {
            Set<Integer> label = dataSet.getMultiLabels()[n].getMatchedLabels();
            labels[n] = new DenseVector(dataSet.getNumClasses());
            for (int l : label) {
                labels[n].set(l, 1);
            }
        }
    }

    public void optimize(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }


    public void iterate(){
        eStep();
        mStep();
        this.terminator.add(getObjective());
    }


    public Terminator getTerminator() {
        return terminator;
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

    /**
     * update all gammas
     */
    private void updateGamma(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    /**
     *
     * @param n data point index
     */
    private void updateGamma(int n){
        Vector feature = dataSet.getRow(n);
        Vector label = this.labels[n];
        int numClusters = bmmClassifier.numClusters;
        double[] logLogisticProbs = bmmClassifier.logisticRegression.predictLogClassProbs(feature);
        double[] logClusterConditionalProbs = bmmClassifier.clusterConditionalLogProbArr(label);
        double[] logNumerators = new double[logLogisticProbs.length];
        for (int k=0;k<numClusters;k++){
            logNumerators[k] = logLogisticProbs[k] + logClusterConditionalProbs[k];
        }
        double logDenominator = MathUtil.logSumExp(logNumerators);
        for (int k=0;k<numClusters;k++){
            gammas[n][k] = Math.exp(logNumerators[k] - logDenominator);
        }
    }

    private double getEntropy(){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i){
        return Entropy.entropy(gammas[i]);
    }

    private double getMStepObjective(){
        //todo add constant
        LogisticLoss logisticLoss = new LogisticLoss(bmmClassifier.logisticRegression,dataSet,
                gammas,gaussianPriorVariance);
        return logisticLoss.getValue() + getEntropy() + bernoulliObj();
    }


    //todo parallel
    private double bernoulliObj(){
        double res =  IntStream.range(0, dataSet.getNumDataPoints())
                .mapToDouble(this::bernoulliObj).sum();
        if (logger.isDebugEnabled()){
            logger.debug("bernoulli objective = "+res);
        }
        return res;
    }

    private double bernoulliObj(int dataPoint){
        double res = 0;
        for (int k=0;k<bmmClassifier.numClusters;k++){
            res += bernoulliObj(dataPoint,k);
        }
        return res;
    }

    private double bernoulliObj(int dataPoint, int cluster){
        if (gammas[dataPoint][cluster]<1E-10){
            return 0;
        }
        double sum = 0;
        BernoulliDistribution[][] distributions = bmmClassifier.distributions;
        int numLabels = dataSet.getNumClasses();
        for (int l=0;l<numLabels;l++){
            double mu = distributions[cluster][l].getP();
            double label = labels[dataPoint].get(l);
            // unstable if compute directly
            if (label==1){
                if (mu==0){
                    throw new RuntimeException("label=1 and mu=0, gamma nk = "+gammas[dataPoint][cluster]);
                }
                sum += Math.log(mu);

            } else {
                // label == 0
                if (mu==1){
                    throw new RuntimeException("label=0 and mu=1, gamma nk"+gammas[dataPoint][cluster]);
                }
                sum += Math.log(1-mu);
            }
        }
        return -1*gammas[dataPoint][cluster]*sum;
    }


    private void mStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start M step");
        }
        updateBernoullis();
        updateLogisticRegression();
        if (logger.isDebugEnabled()){
            logger.debug("finish M step");
            logger.debug("objective = "+getObjective());
        }
    }


    private double getObjective(){

        double mObj = getMStepObjective();
        double entropy = getEntropy();
        double emObj = mObj - entropy;
        if (logger.isDebugEnabled()){
            logger.debug("M step objective = "+mObj);
            logger.debug("entropy = "+entropy);
            logger.debug("EM objective = "+emObj);
        }
        return emObj;
    }

    private void updateBernoullis(){
        IntStream.range(0, bmmClassifier.numClusters).parallel().forEach(this::updateBernoulli);
    }

    private void updateBernoulli(int k) {
        double nk = 0;
        for (int n=0; n<dataSet.getNumDataPoints(); n++){
            nk += gammas[n][k];
        }
        Vector average = new DenseVector(bmmClassifier.getNumClasses());
        for (int n=0; n<dataSet.getNumDataPoints(); n++){
            average = average.plus(labels[n].times(gammas[n][k]));
        }
        average = average.divide(nk);
        for (int l=0; l<bmmClassifier.numLabels; l++){
            bmmClassifier.distributions[k][l] = new BernoulliDistribution(average.get(l));
        }

    }

    private void updateLogisticRegression(){
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(bmmClassifier.logisticRegression, dataSet, gammas, gaussianPriorVariance);
        ridgeLogisticOptimizer.optimize();
    }

    public static BMMOptimizer deserialize(File file) throws Exception {
        try (
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            BMMOptimizer bmmOptimizer = (BMMOptimizer) objectInputStream.readObject();
            return bmmOptimizer;
        }
    }

    public static BMMOptimizer deserialize(String file) throws Exception {
        File file1 = new File(file);
        return deserialize(file1);
    }


    public void serialize(File file) throws Exception {
        File parent = file.getParentFile();
        if (!parent.exists()) {
            parent.mkdir();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    public void serialize(String file) throws Exception {
        File file1 = new File(file);
        serialize(file1);
    }

}
