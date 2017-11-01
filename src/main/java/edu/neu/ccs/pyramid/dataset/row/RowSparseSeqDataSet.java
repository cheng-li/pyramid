package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.Density;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by Rainicy on 8/31/17
 */
public class RowSparseSeqDataSet extends  AbstractRowDataSet implements RowDataSet {
    protected transient SequentialAccessSparseVector[] featureRows;
    /**
     * for each data point i, save its norm2.
     * size of numDataPoints
     */
    protected double[] featureNorm;

    public RowSparseSeqDataSet(int numDatapoints, int numFeatures) {
        super(numDatapoints, numFeatures);
        this.featureRows = new SequentialAccessSparseVector[numDatapoints];
        IntStream.range(0, numDatapoints).parallel().forEach(i -> {
            featureRows[i] = new SequentialAccessSparseVector(numFeatures);
        });
//        for (int i=0; i<numDatapoints; i++) {
//            this.featureRows[i] = new SequentialAccessSparseVector(numFeatures);
//        }
        featureNorm = new double[numDatapoints];
    }

    @Override
    public Vector getRow(int dataPointIndex) {
        return featureRows[dataPointIndex];
    }

    @Override
    public void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        featureRows[dataPointIndex].setQuick(featureIndex, featureValue);
    }

    @Override
    public double[] getNorm() {
        return featureNorm;
    }

    @Override
    public void setNorm(int dataPointIndex, double norm) {
        featureNorm[dataPointIndex] = norm;
    }

    @Override
    public boolean isDense() {
        return false;
    }

    @Override
    public Density density() {
        return Density.SPARSE_SEQUENTIAL;
    }
}

