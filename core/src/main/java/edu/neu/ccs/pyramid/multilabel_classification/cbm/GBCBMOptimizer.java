package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by chengli on 4/29/17.
 */
public class GBCBMOptimizer extends AbstractCBMOptimizer{
    private static final Logger logger = LogManager.getLogger();
    private int numLeaves=2;
    private double shrinkage=1;

    public GBCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
        super(cbm, dataSet);
        this.parallelBinaryUpdates =false;
    }

    public void setNumLeaves(int numLeaves) {
        this.numLeaves = numLeaves;
    }

    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    @Override
    protected void updateBinaryClassifier(int component, int label, MultiLabelClfDataSet activeDataset, double[] activeGammas) {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        if (cbm.binaryClassifiers[component][label] == null || cbm.binaryClassifiers[component][label] instanceof PriorProbClassifier){
            cbm.binaryClassifiers[component][label] = new LKBoost(2);
        }

        int[] binaryLabels = DataSetUtil.toBinaryLabels(activeDataset.getMultiLabels(), label);
        double[][] targetsDistributions = DataSetUtil.labelsToDistributions(binaryLabels, 2);

        LKBoost boost = (LKBoost)this.cbm.binaryClassifiers[component][label];
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeaves);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(2));
        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost,activeDataset, regTreeFactory,
                activeGammas,targetsDistributions);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        optimizer.iterate(binaryUpdatesPerIter);

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
        LKBoost boost = (LKBoost)this.cbm.multiClassClassifier;
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(numLeaves);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(cbm.getNumComponents()));

        LKBoostOptimizer optimizer = new LKBoostOptimizer(boost, dataSet, regTreeFactory, gammas);
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();
        optimizer.iterate(multiclassUpdatesPerIter);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateMultiClassClassifier");
        }
    }

    // todo deal with prior classifier
    @Override
    protected double binaryObj(int component, int classIndex) {
        return 0;
    }

    @Override
    protected double multiClassClassifierObj() {
        return 0;
    }
}
