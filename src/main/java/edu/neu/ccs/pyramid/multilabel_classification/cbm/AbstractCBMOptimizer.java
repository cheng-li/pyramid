package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 3/21/17.
 */
public abstract class AbstractCBMOptimizer {
    private static final Logger logger = LogManager.getLogger();
    protected CBM cbm;
    protected MultiLabelClfDataSet dataSet;

    // format [#data][#components]
    protected double[][] gammas;
    // format [#components][#data]
    protected double[][] gammasT;

    public AbstractCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
        this.cbm = cbm;
        this.dataSet = dataSet;

        this.gammas = new double[dataSet.getNumDataPoints()][cbm.getNumComponents()];
        this.gammasT = new double[cbm.getNumComponents()][dataSet.getNumDataPoints()];
        double average = 1.0/ cbm.getNumComponents();
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            for (int k = 0; k< cbm.getNumComponents(); k++){
                gammas[n][k] = average;
                gammasT[k][n] = average;
            }
        }
    }

    public void iterate() {
        eStep();
        mStep();
    }

    protected void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        updateGamma();
        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+getObjective());
        }
    }


    protected void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    protected void updateGamma(int n) {
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

    protected void updateBinaryClassifiers() {
        if (logger.isDebugEnabled()){
            logger.debug("start updateBinaryClassifiers");
        }
        IntStream.range(0, cbm.numComponents).forEach(this::updateBinaryClassifiers);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateBinaryClassifiers");
        }
    }

    //todo pay attention to parallelism
    protected void updateBinaryClassifiers(int component){
        IntStream.range(0, cbm.numLabels).parallel().forEach(l-> updateBinaryClassifier(component,l));
    }

    abstract protected void updateBinaryClassifier(int component, int label);


    protected abstract  void updateMultiClassClassifier();



    //******************** for debugging *****************************

    //TODO: have to modify the objectives by introducing L1 regularization part
    public double getObjective() {
        return multiClassClassifierObj() + binaryObj();
    }

    protected double binaryObj(){
        return IntStream.range(0, cbm.numComponents).mapToDouble(this::binaryObj).sum();
    }

    protected double binaryObj(int component){
        return IntStream.range(0, cbm.numLabels).parallel().mapToDouble(l->binaryObj(component,l)).sum();
    }

    protected abstract double binaryObj(int clusterIndex, int classIndex);

    protected abstract double multiClassClassifierObj();

    public double[][] getGammas() {
        return gammas;
    }


}
