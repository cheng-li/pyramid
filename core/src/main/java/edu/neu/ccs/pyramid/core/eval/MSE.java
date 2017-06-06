package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/6/14.
 */
public class MSE {
    public static double mse(double[] labels, double[] predictions){
        if (labels.length != predictions.length){
            throw new IllegalArgumentException("dimensions don't match");
        }
        double squaredError = 0;
        for (int i=0;i<labels.length;i++){
            squaredError += Math.pow(labels[i]-predictions[i],2);
        }
        return squaredError/labels.length;
    }

    /**
     * parallel
     * @param regressor
     * @param dataSet
     * @return
     */
    public static double mse(Regressor regressor, RegDataSet dataSet){
        int numDataPoints = dataSet.getNumDataPoints();
        double[] labels = dataSet.getLabels();
        double result = IntStream.range(0, numDataPoints).parallel()
                .mapToDouble(i -> Math.pow(regressor.predict(dataSet.getRow(i))-labels[i],2))
                .average().getAsDouble();
        return result;
    }

    public static double mseForNumLabels(MultiLabelClassifier multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet){
        double[] truth = Arrays.stream(multiLabelClfDataSet.getMultiLabels()).mapToDouble(MultiLabel::getNumMatchedLabels).toArray();
        double[] predi = IntStream.range(0,multiLabelClfDataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->multiLabelClassifier.predict(multiLabelClfDataSet.getRow(i)).getNumMatchedLabels())
                .toArray();
        return mse(truth,predi);

    }
}
