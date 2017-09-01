package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.Density;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 8/31/17
 */
abstract class AbstractRowDataSet implements RowDataSet {
    protected int numDataPoints;
    protected int numFeatures;

    public AbstractRowDataSet(int numDatapoints, int numFeatures) {
        this.numDataPoints = numDatapoints;
        this.numFeatures = numFeatures;
    }

    @Override
    public int getNumDataPoints() {
        return numDataPoints;
    }

    @Override
    public int getNumFeatures() {
        return numFeatures;
    }

    @Override
    public abstract void setNorm(int dataPointIndex, double norm);

    @Override
    public abstract  double[] getNorm();

    @Override
    public abstract Vector getRow(int dataPointIndex);

    @Override
    public abstract void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue);

    @Override
    public abstract boolean isDense();

    @Override
    public abstract Density density();

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of data points = ").append(numDataPoints).append("\n");
        sb.append("number of features = ").append(numFeatures).append("\n");
        sb.append("row matrix:").append("\n");
        for (int i=0;i<numDataPoints;i++) {
            sb.append(i).append(":\t").append(getRow(i).asFormatString()).append("\n");
        }
        return sb.toString();
    }
}
