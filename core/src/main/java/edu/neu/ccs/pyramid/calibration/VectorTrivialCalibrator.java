package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VectorTrivialCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    private double average;


    public VectorTrivialCalibrator(RegDataSet regDataSet) {
         average = 1.0*MathUtil.arraySum(regDataSet.getLabels())/regDataSet.getNumDataPoints();

    }
    @Override
    public double calibrate(Vector vector) {
        return average;
    }
}
