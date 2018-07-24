package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.ArgMin;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VectorCardSetCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    Map<Integer,IsotonicRegression> calibrations;
    private int scoreIndex;
    private int cardIndex;

    public VectorCardSetCalibrator(ClfDataSet clfDataSet, int scoreIndex, int cardIndex) {
        this.scoreIndex = scoreIndex;
        this.cardIndex = cardIndex;
        this.calibrations = new HashMap<>();
        Set<Integer> cardinalities = new HashSet<>();
        for (int i=0;i<clfDataSet.getNumDataPoints();i++) {
            cardinalities.add((int)clfDataSet.getRow(i).get(cardIndex));
        }

        for (int cardinality : cardinalities) {
            Stream<Pair<Double, Integer>> stream = IntStream.range(0, clfDataSet.getNumDataPoints()).parallel()
                    .boxed().filter(i->((int)clfDataSet.getRow(i).get(cardIndex))==cardinality)
                    .map(i->new Pair<>(clfDataSet.getRow(i).get(scoreIndex),clfDataSet.getLabels()[i]));
            calibrations.put(cardinality, new IsotonicRegression(stream));
        }
    }


    public double calibrate(Vector vector){
        double uncalibrated = vector.get(scoreIndex);
        int cardinality = (int)vector.get(cardIndex);
        //deal with unseen cardinality
        List<Integer> cards = new ArrayList<>(calibrations.keySet());
        Collections.sort(cards);
        double[] diff = new double[cards.size()];
        for (int i=0;i<cards.size();i++){
            diff[i] = Math.abs(cardinality-cards.get(i));
        }
        int closest = cards.get(ArgMin.argMin(diff));
        return calibrations.get(closest).predict(uncalibrated);
    }
}
