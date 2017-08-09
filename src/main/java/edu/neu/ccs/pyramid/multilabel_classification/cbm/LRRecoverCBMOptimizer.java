package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by chengli on 4/9/17.
 */
public class LRRecoverCBMOptimizer extends AbstractRecoverCBMOptimizer{
    private static final Logger logger = LogManager.getLogger();
    // regularization for multiClassClassifier
    private double priorVarianceMultiClass =1;
    // regularization for binary logisticRegression
    private double priorVarianceBinary =1;

    public LRRecoverCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
        super(cbm, dataSet);
    }


    public void setPriorVarianceMultiClass(double priorVarianceMultiClass) {
        this.priorVarianceMultiClass = priorVarianceMultiClass;
    }

    public void setPriorVarianceBinary(double priorVarianceBinary) {
        this.priorVarianceBinary = priorVarianceBinary;
    }

    @Override
    protected void updateBinaryClassifier(int component, int label, MultiLabelClfDataSet activeDataset, double[] activeGammas) {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        if (cbm.binaryClassifiers[component][label] == null || cbm.binaryClassifiers[component][label] instanceof PriorProbClassifier){
            cbm.binaryClassifiers[component][label] = new LogisticRegression(2, activeDataset.getNumFeatures());
        }

        RidgeLogisticOptimizer ridgeLogisticOptimizer;

        int[] binaryLabels = DataSetUtil.toBinaryLabels(activeDataset.getMultiLabels(), label);
        // no parallelism
        ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbm.binaryClassifiers[component][label],
                activeDataset, binaryLabels, activeGammas, priorVarianceBinary, false);

        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(binaryUpdatesPerIter);
        ridgeLogisticOptimizer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("time spent on updating component "+component+" label "+label+" = "+stopWatch);
        }
    }

    @Override
    protected void updateMultiClassClassifier() {
        if (logger.isDebugEnabled()){
            logger.debug("start updateMultiClassClassifier");
        }
        // parallel
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer((LogisticRegression)cbm.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        ridgeLogisticOptimizer.getOptimizer().getTerminator().setMaxIteration(multiclassUpdatesPerIter);
        ridgeLogisticOptimizer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("finish updateMultiClassClassifier");
        }
    }

    // todo deal with prior classifier
    @Override
    protected double binaryObj(int component, int classIndex) {
        int[] binaryLabels = DataSetUtil.toBinaryLabels(groundTruth.getMultiLabels(), classIndex);
        double[][] targetsDistribution = DataSetUtil.labelsToDistributions(binaryLabels, 2);
        double[] weights = new double[groundTruth.getNumDataPoints()];
        for (int i=0;i<groundTruth.getNumDataPoints();i++){
            weights[i] = gammas[i][component];
        }
        LogisticLoss logisticLoss = new LogisticLoss((LogisticRegression) cbm.binaryClassifiers[component][classIndex],
                groundTruth, weights, targetsDistribution, priorVarianceBinary, false);
        return logisticLoss.getValue();
    }

    @Override
    protected double multiClassClassifierObj() {
        LogisticLoss logisticLoss =  new LogisticLoss((LogisticRegression) cbm.multiClassClassifier,
                dataSet, gammas, priorVarianceMultiClass, true);
        return logisticLoss.getValue();
    }
}
