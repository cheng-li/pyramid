package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.calibration.VectorCalibrator;
import org.apache.mahout.math.Vector;

public class VectorPlattScaling implements VectorCalibrator {
    private int scoreIndex;
    private PlattScaling plattScaling;

    public VectorPlattScaling(ClfDataSet clfDataSet, int scoreIndex) {
        this.scoreIndex = scoreIndex;
        double[] scores = new double[clfDataSet.getNumDataPoints()];
        for (int i=0;i<clfDataSet.getNumDataPoints();i++){
            scores[i] = clfDataSet.getRow(i).get(scoreIndex);
        }
        int[] labels = clfDataSet.getLabels();
        this.plattScaling = new PlattScaling(scores, labels);
    }

    @Override
    public double calibrate(Vector vector) {
        return plattScaling.transform(vector.get(scoreIndex));
    }
}
