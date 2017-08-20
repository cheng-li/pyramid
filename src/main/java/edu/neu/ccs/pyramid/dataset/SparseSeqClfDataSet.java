package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * featureRows for dot product
 * if feature columns is used for more get, change it to RandomAccessSparseVector later.
 */
public class SparseSeqClfDataSet extends AbstractDataSet implements ClfDataSet {

    protected transient SequentialAccessSparseVector[] featureRows;
    protected transient SequentialAccessSparseVector[] featureColumns;

    int numClasses;
    private int[] labels;
    private LabelTranslator labelTranslator;

    public SparseSeqClfDataSet(int numDataPoints, int numFeatures, boolean missingValue, int numClasses) {
        super(numDataPoints, numFeatures, missingValue);
        this.featureRows = new SequentialAccessSparseVector[numDataPoints];
        IntStream.range(0, numDataPoints).parallel().forEach(i -> {
            this.featureRows[i] = new SequentialAccessSparseVector(numFeatures);
        });
        // may be changed to RandomAccessSparseVector later
        this.featureColumns = new SequentialAccessSparseVector[numFeatures];
        IntStream.range(0, numFeatures).parallel().forEach(j ->{
            this.featureColumns[j] = new SequentialAccessSparseVector(numDataPoints);
        });

        this.labels = new int[numDataPoints];
        this.numClasses = numClasses;
        this.labelTranslator = LabelTranslator.newDefaultLabelTranslator(numClasses);
    }

    @Override
    public Vector getColumn(int featureIndex) {
        return this.featureColumns[featureIndex];
    }

    @Override
    public Vector getRow(int dataPointIndex) {
        return this.featureRows[dataPointIndex];
    }

    @Override
    public void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        if ((!this.hasMissingValue()) && Double.isNaN(featureValue)){
            throw new IllegalArgumentException("missing value is not allowed in this data set");
        }
        this.featureRows[dataPointIndex].set(featureIndex, featureValue);
        this.featureColumns[featureIndex].set(dataPointIndex, featureValue);
    }

    @Override
    public boolean isDense() {
        return false;
    }

    @Override
    public Density density() {
        return Density.SPARSE_SEQUENTIAL;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public int[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, int label) {
        if (label<0||label>=this.numClasses){
            throw new IllegalArgumentException("label<0||label>=this.numClasses");
        }
        this.labels[dataPointIndex]=label;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    @Override
    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
