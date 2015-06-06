package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.Regressor;

/**
 * Created by chengli on 6/5/15.
 */
public class RMSE {
    public static double rmse(double[] labels, double[] predictions){
        return Math.pow(MSE.mse(labels,predictions),0.5);
    }

    public static double rmse(Regressor regressor, RegDataSet dataSet){
        return Math.pow(MSE.mse(regressor,dataSet),0.5);
    }
}
