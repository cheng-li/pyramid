package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.Density;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

/**
 * Created by Rainicy on 8/31/17
 *
 * Re-implement the DataSet, which removes the column feature and adds feature norm2 per instance.
 */
public interface RowDataSet extends Serializable {
    int getNumDataPoints();
    int getNumFeatures();
    Vector getRow(int dataPointIndex);
    void setFeatureValue(int dataPointIndex,
                         int featureIndex, double featureValue);
    double[] getNorm();
    void setNorm(int dataPointIndex, double norm);
    boolean isDense();
    Density density();
}
