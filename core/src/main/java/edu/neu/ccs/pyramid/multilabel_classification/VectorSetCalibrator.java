package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.VectorCalibrator;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VectorSetCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private int scoreIndex;

    public VectorSetCalibrator(ClfDataSet clfDataSet, int scoreIndex) {
        this.scoreIndex = scoreIndex;
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, clfDataSet.getNumDataPoints())
                .mapToObj(i->new Pair<>(clfDataSet.getRow(i).get(scoreIndex),clfDataSet.getLabels()[i]));
        isotonicRegression = new IsotonicRegression(stream);

    }

    public double calibrate(Vector vector){
        return isotonicRegression.predict(vector.get(scoreIndex));
    }

}
