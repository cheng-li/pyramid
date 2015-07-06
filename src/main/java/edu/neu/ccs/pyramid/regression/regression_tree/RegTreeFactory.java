package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;

/**
 * Created by chengli on 7/5/15.
 */
public class RegTreeFactory implements RegressorFactory {
    private RegTreeConfig regTreeConfig;

    public RegTreeFactory(RegTreeConfig regTreeConfig) {
        this.regTreeConfig = regTreeConfig;
    }

    @Override
    public Regressor fit(DataSet dataSet, double[] labels) {
        return RegTreeTrainer.fit(regTreeConfig,dataSet,labels);
    }
}
