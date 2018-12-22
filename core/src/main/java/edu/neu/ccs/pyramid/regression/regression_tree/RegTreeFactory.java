package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;

/**
 * Created by chengli on 7/5/15.
 */
public class RegTreeFactory implements RegressorFactory {
    private RegTreeConfig regTreeConfig;
    private LeafOutputCalculator leafOutputCalculator;

    public RegTreeFactory(RegTreeConfig regTreeConfig) {
        this.regTreeConfig = regTreeConfig;
        this.leafOutputCalculator = new AverageOutputCalculator();
    }

    public void setLeafOutputCalculator(LeafOutputCalculator leafOutputCalculator) {
        this.leafOutputCalculator = leafOutputCalculator;
    }

    @Override
    public Regressor fit(DataSet dataSet, double[] labels) {
        return RegTreeTrainer.fit(regTreeConfig,dataSet,labels,leafOutputCalculator);
    }

    @Override
    public Regressor fit(DataSet dataSet, double[] labels, double[] weights) {
        return RegTreeTrainer.fit(regTreeConfig,dataSet,labels,weights, leafOutputCalculator);
    }


    public Regressor fit(DataSet dataSet, double[] labels, double[] weights, int[] monotonicity) {
        return RegTreeTrainer.fit(regTreeConfig,dataSet,labels,weights, leafOutputCalculator, monotonicity);
    }
}
