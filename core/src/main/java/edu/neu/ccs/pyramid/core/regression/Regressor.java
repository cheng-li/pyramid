package edu.neu.ccs.pyramid.core.regression;


import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.dataset.DataSet;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/6/14.
 */
public interface Regressor extends Serializable {
    double predict(Vector vector);
    FeatureList getFeatureList();
    default double[] predict(DataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                mapToDouble(i -> predict(dataSet.getRow(i))).toArray();
    }
}
