package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.clustering.bm.BM;
import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
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


    // if the fraction of positive labels < threshold, or > 1-threshold,  skip the binary model, use prior probability
    // set threshold = 0 if we don't want to skip any
    protected double skipLabelThreshold = 1E-5;

    // if gamma_i^k is smaller than this threshold, skip it when training binary classifiers in component k
    // set threshold = 0 if we don't want to skip any
    protected double skipDataThreshold = 1E-5;


    protected int multiclassUpdatesPerIter = 20;
    protected int binaryUpdatesPerIter = 20;

    protected DataSet labelMatrix;

    public AbstractCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
        this.cbm = cbm;
        this.dataSet = dataSet;

        this.gammas = new double[dataSet.getNumDataPoints()][cbm.getNumComponents()];
        double average = 1.0/ cbm.getNumComponents();
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            for (int k = 0; k< cbm.getNumComponents(); k++){
                gammas[n][k] = average;
            }
        }
        this.labelMatrix = DataSetBuilder.getBuilder()
                .numDataPoints(dataSet.getNumDataPoints())
                .numFeatures(dataSet.getNumClasses())
                .density(Density.SPARSE_RANDOM)
                .build();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l: multiLabel.getMatchedLabels()){
                labelMatrix.setFeatureValue(i,l,1);
            }
        }
    }

    public void setMulticlassUpdatesPerIter(int multiclassUpdatesPerIter) {
        this.multiclassUpdatesPerIter = multiclassUpdatesPerIter;
    }

    public void setBinaryUpdatesPerIter(int binaryUpdatesPerIter) {
        this.binaryUpdatesPerIter = binaryUpdatesPerIter;
    }

    public void setSkipLabelThreshold(double skipLabelThreshold) {
        this.skipLabelThreshold = skipLabelThreshold;
    }

    public void setSkipDataThreshold(double skipDataThreshold) {
        this.skipDataThreshold = skipDataThreshold;
    }

    public void initialize(){
        gammas = BMSelector.selectGammas(dataSet.getNumClasses(),dataSet.getMultiLabels(), cbm.getNumComponents());
        System.out.println("performing M step");
        mStep();
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
//            logger.debug("objective = "+getObjective());
        }
    }


    protected void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    protected void updateGamma(int n) {
        Vector x = dataSet.getRow(n);
        MultiLabel y = dataSet.getMultiLabels()[n];
        double[] posterior = cbm.posteriorMembershipShortCircuit(x, y);
        for (int k=0; k<cbm.numComponents; k++) {
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
//            logger.debug("objective = "+getObjective());
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

        if (logger.isDebugEnabled()){
            logger.debug("computing active dataset for component " +component);
        }

        // skip small gammas
        List<Double> activeGammasList = new ArrayList<>();
        List<Integer> activeIndices = new ArrayList<>();
        double weightedTotal = 0;
        double thresholdedWeightedTotal = 0;
        int counter = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            double v = gammas[i][component];
            weightedTotal += v;
            if (v>= skipDataThreshold){
                activeGammasList.add(v);
                activeIndices.add(i);
                thresholdedWeightedTotal += v;
                counter += 1;
            }
        }

        double[] activeGammas = activeGammasList.stream().mapToDouble(a->a).toArray();

        if (logger.isDebugEnabled()){
            logger.debug("number of active data  = "+ counter);
            logger.debug("total weight  = "+weightedTotal);
            logger.debug("total weight of active data  = "+thresholdedWeightedTotal);
            logger.debug("creating active dataset");
        }


        MultiLabelClfDataSet activeDataSet = DataSetUtil.sampleData(dataSet, activeIndices);
        int activeFeatures = (int) IntStream.range(0, activeDataSet.getNumFeatures()).filter(j->activeDataSet.getColumn(j).getNumNonZeroElements()>0).count();
        if (logger.isDebugEnabled()){
            logger.debug("active dataset created");
            logger.debug("number of active features = "+activeFeatures);
        }

        // to please lambda
        final double totalWeight = weightedTotal;
        IntStream.range(0, cbm.numLabels).parallel()
                .forEach(l-> skipOrUpdateBinaryClassifier(component,l, activeDataSet, activeGammas, totalWeight));
    }


    protected void skipOrUpdateBinaryClassifier(int component, int label, MultiLabelClfDataSet activeDataSet,
                                                double[] activeGammas, double totalWeight){
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        double effectivePositives = effectivePositives(component, label);
        double positiveProb = effectivePositives/totalWeight;

        StringBuilder sb = new StringBuilder();
        sb.append("for component ").append(component).append(", label ").append(label);
        sb.append(", weighted positives = ").append(effectivePositives);
        sb.append(", positive fraction = "+positiveProb);


//        if (positiveProb==0){
//            positiveProb=1.0E-50;
//        }

        // it be happen that p >1 for numerical reasons
        if (positiveProb>=1){
            positiveProb=1;
        }

        if (positiveProb<skipLabelThreshold || positiveProb>1-skipLabelThreshold){
            double[] probs = {1-positiveProb, positiveProb};
            cbm.binaryClassifiers[component][label] = new PriorProbClassifier(probs);
            sb.append(", skip, use prior = ").append(positiveProb);
            sb.append(", time spent = ").append(stopWatch.toString());
            if (logger.isDebugEnabled()){
                logger.debug(sb.toString());
            }
            return;
        }

        if (logger.isDebugEnabled()){
            logger.debug(sb.toString());
        }
        updateBinaryClassifier(component, label, activeDataSet, activeGammas);
    }

    abstract protected void updateBinaryClassifier(int component, int label, MultiLabelClfDataSet activeDataset, double[] activeGammas);

    protected abstract void updateMultiClassClassifier();

    private double effectivePositives(int componentIndex, int labelIndex){
        double sum = 0;
        Vector labelColumn = labelMatrix.getColumn(labelIndex);
        for (Vector.Element element: labelColumn.nonZeroes()){
            int dataIndex = element.index();
            sum += gammas[dataIndex][componentIndex];
        }
        return sum;
    }



    //******************** for debugging *****************************

    public double getObjective() {
        return multiClassClassifierObj() + binaryObj();
    }

    protected double binaryObj(){
        return IntStream.range(0, cbm.numComponents).mapToDouble(this::binaryObj).sum();
    }

    protected double binaryObj(int component){
        return IntStream.range(0, cbm.numLabels).parallel().mapToDouble(l->binaryObj(component,l)).sum();
    }

    protected abstract double binaryObj(int component, int classIndex);

    protected abstract double multiClassClassifierObj();

    public double[][] getGammas() {
        return gammas;
    }

    private void checkGamma(){
        for (int i=0;i<gammas.length;i++){
            for (int k=0;k<gammas[0].length;k++){
                if (Double.isNaN(gammas[i][k])){
                    throw new RuntimeException("gamma "+i+" "+k+" is NaN");
                }
            }
        }
    }


}
