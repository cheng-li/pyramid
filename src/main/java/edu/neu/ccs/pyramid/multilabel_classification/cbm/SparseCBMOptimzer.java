package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.clustering.bm.BM;
import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;


/**
 * Created by chengli on 3/5/17.
 */
public class SparseCBMOptimzer {
    private CBM cbm;
    private MultiLabelClfDataSet dataSet;
    // format [#data][#components]
    double[][] gammas;
    BM bm;
    // regularization for multiClassClassifier
    private double priorVarianceMultiClass =1;
    // regularization for binary logisticRegression
    private double priorVarianceBinary =1;

    //todo init lr with component conditional prior


    public SparseCBMOptimzer(CBM cbm, MultiLabelClfDataSet dataSet) {
        this.cbm = cbm;
        this.dataSet = dataSet;
    }

    public void setPriorVarianceMultiClass(double priorVarianceMultiClass) {
        this.priorVarianceMultiClass = priorVarianceMultiClass;
    }

    public void setPriorVarianceBinary(double priorVarianceBinary) {
        this.priorVarianceBinary = priorVarianceBinary;
    }

    void initalizeGammaByBM(){
        Pair<BM, double[][]> pair = BMSelector.selectAll(dataSet.getNumClasses(),dataSet.getMultiLabels(), cbm.getNumComponents());
        bm = pair.getFirst();
        gammas = pair.getSecond();
    }

    public void updateMultiClassLR() {
        // parallel
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbm.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        //TODO maximum iterations
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(50);
        ridgeLogisticOptimizer.optimize();
    }

    public void updateAllBinary(){
        for (int k=0;k<cbm.getNumComponents();k++){
            for (int l=0;l<cbm.getNumClasses();l++){
                updateBinaryLogisticRegression(k,l);
            }
        }
    }

    private void updateBinaryLogisticRegression(int componentIndex, int labelIndex){
        System.out.println("--------------------");
        System.out.println("for component "+componentIndex+", label "+labelIndex);
        double effectivePositives = effectivePositives(componentIndex, labelIndex);
        System.out.println("effective positives = "+effectivePositives);
        if (effectivePositives<=1){
            double positiveProb = prior(componentIndex, labelIndex);
            double[] probs = {1-positiveProb, positiveProb};
            cbm.binaryClassifiers[componentIndex][labelIndex] = new LogisticRegression(2, dataSet.getNumFeatures(), probs);
            System.out.println("skip, use prior = "+positiveProb);
            return;
        }


        RidgeLogisticOptimizer ridgeLogisticOptimizer;
        double[] instanceWeights = new double[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            //todo do not multiply by noiseSetWeight ?
            instanceWeights[i] = gammas[i][componentIndex];
        }
        int[] binaryLabels = DataSetUtil.toBinaryLabels(dataSet.getMultiLabels(), labelIndex);
        // no parallelism
        ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbm.binaryClassifiers[componentIndex][labelIndex],
                dataSet, binaryLabels, instanceWeights, priorVarianceBinary, true);
        //TODO maximum iterations
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(50);
        ridgeLogisticOptimizer.optimize();
    }

    private double effectivePositives(int componentIndex, int labelIndex){
        double sum = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            if (dataSet.getMultiLabels()[i].matchClass(labelIndex)){
                sum += gammas[i][componentIndex];
            }
        }
        return sum;
    }


    public void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    private void updateGamma(int n) {
        Vector x = dataSet.getRow(n);
        MultiLabel y = dataSet.getMultiLabels()[n];
        double[] posterior = cbm.posteriorMembership(x, y);
        for (int k=0; k<cbm.numComponents; k++) {
            gammas[n][k] = posterior[k];
        }
    }

    private double prior(int componentIndex, int labelIndex){
        double positives = 0;
        double total = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            total += gammas[i][componentIndex];
            if (dataSet.getMultiLabels()[i].matchClass(labelIndex)){
                positives += gammas[i][componentIndex];
            }
        }
        return positives/total;
    }
}
