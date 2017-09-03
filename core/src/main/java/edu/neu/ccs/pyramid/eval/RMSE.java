package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.Arrays;
import java.util.stream.IntStream;

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


    public static double rmseForNumLabels(MultiLabelClassifier multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet){
        return Math.pow(MSE.mseForNumLabels(multiLabelClassifier,multiLabelClfDataSet),0.5);

    }

    public static double normalizedRMSE(double[] labels, double[] predictions){
        double rmse =rmse(labels, predictions);
        double max = Arrays.stream(labels).max().getAsDouble();
        double min = Arrays.stream(labels).min().getAsDouble();
        return rmse/(max-min);
    }
}
