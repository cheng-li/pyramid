package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.util.ArgMax;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/7/17.
 */
public abstract class AbstractRobustCBMOptimizer {

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


    protected int multiclassUpdatesPerIter = 10;
    protected int binaryUpdatesPerIter = 10;

    protected double smoothingStrength =0.0001;

    // number of positives for all labels
    private int[] positiveCounts;

    protected DataSet labelMatrix;


    private double noiseGammaLabel = 0;
    private double[][] noiseLabelWeights;
    private double[][] marginals;


    public AbstractRobustCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
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

        positiveCounts = new int[dataSet.getNumClasses()];
        for (int l=0;l<dataSet.getNumClasses();l++){
            positiveCounts[l] = labelMatrix.getColumn(l).getNumNonZeroElements();
        }

        this.marginals = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        this.noiseLabelWeights = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            Arrays.fill(noiseLabelWeights[i],1);
        }

    }

    public void setNoiseGammaLabel(double noiseGammaLabel) {
        this.noiseGammaLabel = noiseGammaLabel;
    }

    public double[][] getNoiseLabelWeights() {
        return noiseLabelWeights;
    }

    public double[][] getMarginals() {
        return marginals;
    }

    public void setSmoothingStrength(double smoothingStrength) {
        this.smoothingStrength = smoothingStrength;
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
        if (logger.isDebugEnabled()){
            logger.debug("performing M step");
        }
        mStep();
    }

    public void iterateSimple(){
        eStep();
        mStep();
    }

    public void iterate() {
        updateMarginals();
        updateLabelWeights();
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
        double[] posterior = cbm.posteriorMembership(x, y, noiseLabelWeights[n]);
        for (int k=0; k<cbm.numComponents; k++) {
            gammas[n][k] = posterior[k];
        }
    }

    private void updateMarginals(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> {
                    double[] predictedMarginals = cbm.predictClassProbs(dataSet.getRow(i));
                    for (int l=0;l<dataSet.getNumClasses();l++){
                        if (dataSet.getMultiLabels()[i].matchClass(l)){
                            marginals[i][l] = predictedMarginals[l];
                        } else {
                            marginals[i][l] = 1- predictedMarginals[l];
                        }
                    }
                });
    }

    private void updateLabelWeights(){
        double total = 0;
        double n = dataSet.getNumDataPoints();
        double numLabels = dataSet.getNumClasses();
        double numEntries = 0;


        for (int i=0;i<n;i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l=0;l<numLabels;l++){
                if (!multiLabel.matchClass(l)){
                    total += Math.pow(marginals[i][l], noiseGammaLabel);
                    numEntries += 1;
                }

            }
        }

        for (int i=0;i<n;i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l=0;l<numLabels;l++){
                if (!multiLabel.matchClass(l)){
                    noiseLabelWeights[i][l] = numEntries*Math.pow(marginals[i][l],noiseGammaLabel)/total;
                }

            }
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
        List<Integer> activeIndices = new ArrayList<>();
        double[] gammasForComponent = IntStream.range(0, dataSet.getNumDataPoints()).mapToDouble(i->gammas[i][component]).toArray();
        int maxIndex = ArgMax.argMax(gammasForComponent);

        double weightedTotal = 0;
        double thresholdedWeightedTotal = 0;
        int counter = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            double v = gammas[i][component];
            weightedTotal += v;
            if (v>= skipDataThreshold || i==maxIndex){
                activeIndices.add(i);
                thresholdedWeightedTotal += v;
                counter += 1;
            }
        }


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
                .forEach(l-> skipOrUpdateBinaryClassifier(component,l, activeIndices, activeDataSet,  totalWeight));
    }


    protected void skipOrUpdateBinaryClassifier(int component, int label, List<Integer> activeIndices,
                                                MultiLabelClfDataSet activeDataSet,
                                                double totalWeight){
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        double effectivePositives = effectivePositives(component, label);

        double nonSmoothedPositiveProb = effectivePositives/totalWeight;

        // smooth the component-wise label fraction with global label fraction

        double smoothedPositiveProb = (effectivePositives+smoothingStrength*positiveCounts[label])/(totalWeight+smoothingStrength*dataSet.getNumDataPoints());

        StringBuilder sb = new StringBuilder();
        sb.append("for component ").append(component).append(", label ").append(label);
        sb.append(", weighted positives = ").append(effectivePositives);
        sb.append(", non-smoothed positive fraction = "+(effectivePositives/totalWeight));
        sb.append(", global positive fraction = "+((double)positiveCounts[label]/dataSet.getNumDataPoints()));
        sb.append(", smoothed positive fraction = "+smoothedPositiveProb);


//        if (positiveProb==0){
//            positiveProb=1.0E-50;
//        }

        // it be happen that p >1 for numerical reasons
        if (smoothedPositiveProb>=1){
            smoothedPositiveProb=1;
        }

        if (nonSmoothedPositiveProb<skipLabelThreshold || nonSmoothedPositiveProb>1-skipLabelThreshold){
            double[] probs = {1-smoothedPositiveProb, smoothedPositiveProb};
            cbm.binaryClassifiers[component][label] = new PriorProbClassifier(probs);
            sb.append(", skip, use prior = ").append(smoothedPositiveProb);
            sb.append(", time spent = ").append(stopWatch.toString());
            if (logger.isDebugEnabled()){
                logger.debug(sb.toString());
            }
            return;
        }

        if (logger.isDebugEnabled()){
            logger.debug(sb.toString());
        }

        double[] activeInstanceWeights = activeIndices.stream().mapToDouble(i->gammas[i][component]*noiseLabelWeights[i][label]).toArray();
        updateBinaryClassifier(component, label, activeDataSet, activeInstanceWeights);
    }

    abstract protected void updateBinaryClassifier(int component, int label, MultiLabelClfDataSet activeDataset, double[] activeInstanceWeights);

    protected abstract void updateMultiClassClassifier();

    private double effectivePositives(int componentIndex, int labelIndex){
        double sum = 0;
        Vector labelColumn = labelMatrix.getColumn(labelIndex);
        for (Vector.Element element: labelColumn.nonZeroes()){
            int dataIndex = element.index();
            sum += gammas[dataIndex][componentIndex] * noiseLabelWeights[dataIndex][labelIndex];
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
