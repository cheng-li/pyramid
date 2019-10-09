package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.calibration.VectorCalibrator;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;


import java.io.Serializable;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VectorIsoSetCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private int scoreIndex;

//    public VectorIsoSetCalibrator(ClfDataSet clfDataSet, int scoreIndex) {
//        this.scoreIndex = scoreIndex;
//        Stream<Pair<Double, Double>> stream = IntStream.range(0, clfDataSet.getNumDataPoints())
//                .mapToObj(i->new Pair<Double,Double>(clfDataSet.getRow(i).get(scoreIndex),(double)clfDataSet.getLabels()[i]));
//        isotonicRegression = new IsotonicRegression(stream);
//
//    }

    public VectorIsoSetCalibrator(RegDataSet regDataSet, int scoreIndex, boolean interpolate) {
        this.scoreIndex = scoreIndex;
        Stream<Pair<Double, Double>> stream = IntStream.range(0, regDataSet.getNumDataPoints())
                //todo deal with regression labels
                .mapToObj(i->new Pair<>(regDataSet.getRow(i).get(scoreIndex),regDataSet.getLabels()[i]));
        isotonicRegression = new IsotonicRegression(stream, interpolate);
    }

    public double calibrate(Vector vector){
        return isotonicRegression.predict(vector.get(scoreIndex));
    }


    public IsotonicRegression getIsotonicRegression() {
        return isotonicRegression;
    }
}
