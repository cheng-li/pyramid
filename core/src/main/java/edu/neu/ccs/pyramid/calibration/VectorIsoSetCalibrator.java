package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.calibration.VectorCalibrator;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VectorIsoSetCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private int scoreIndex;

    public VectorIsoSetCalibrator(ClfDataSet clfDataSet, int scoreIndex) {
        this.scoreIndex = scoreIndex;
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, clfDataSet.getNumDataPoints())
                .mapToObj(i->new Pair<>(clfDataSet.getRow(i).get(scoreIndex),clfDataSet.getLabels()[i]));
        isotonicRegression = new IsotonicRegression(stream);

    }

    public VectorIsoSetCalibrator(RegDataSet regDataSet, int scoreIndex) {
        this.scoreIndex = scoreIndex;
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, regDataSet.getNumDataPoints())
                //todo deal with regression labels
                .mapToObj(i->new Pair<>(regDataSet.getRow(i).get(scoreIndex),(int)regDataSet.getLabels()[i]));
        isotonicRegression = new IsotonicRegression(stream);
    }

    public double calibrate(Vector vector){
        return isotonicRegression.predict(vector.get(scoreIndex));
    }

}
