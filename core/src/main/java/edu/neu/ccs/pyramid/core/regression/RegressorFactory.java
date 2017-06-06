package edu.neu.ccs.pyramid.core.regression;

import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;

import java.util.Arrays;

/**
 * Created by chengli on 7/5/15.
 */
public interface RegressorFactory {
    Regressor fit(DataSet dataSet, double[] labels, double[] weights);

    default Regressor fit(DataSet dataSet, double[] labels){
        double[] weights = new double[labels.length];
        Arrays.fill(weights,1.0);
        return fit(dataSet,labels,weights);
    }

    default Regressor fit(RegDataSet dataSet){
        return fit(dataSet, dataSet.getLabels());
    }
}
