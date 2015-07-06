package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;

/**
 * Created by chengli on 7/5/15.
 */
public interface RegressorFactory {
    Regressor fit(DataSet dataSet, double[] labels);
    default Regressor fit(RegDataSet dataSet){
        return fit(dataSet, dataSet.getLabels());
    }
}
