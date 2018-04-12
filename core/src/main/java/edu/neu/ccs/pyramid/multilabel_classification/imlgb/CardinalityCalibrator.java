package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.calibration.StreamGenerator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.ArgMin;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CardinalityCalibrator implements Regressor{
    private static final long serialVersionUID = 1L;
    Map<Integer,IsotonicRegression> calibrations;
    private IMLGradientBoosting boosting;


    private CardinalityCalibrator() {
    }

    public CardinalityCalibrator(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        this.calibrations = new HashMap<>();
        List<MultiLabel> support = boosting.getAssignments();
        Set<Integer> cardinalities = new HashSet<>();
        for (MultiLabel multiLabel: support){
            cardinalities.add(multiLabel.getNumMatchedLabels());
        }

        for (int cardinality: cardinalities){
            List<MultiLabel> allAssignments = boosting.getAssignments();
            Stream<Pair<Double,Integer>> stream =  IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .boxed().flatMap(i-> {
                        double[] probs = boosting.predictAllAssignmentProbsWithConstraint(multiLabelClfDataSet.getRow(i));
                        Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length)
                                .filter(a->allAssignments.get(a).getNumMatchedLabels()==cardinality)
                                .mapToObj(a -> {
                            Pair<Double, Integer> pair = new Pair<>();
                            pair.setFirst(probs[a]);
                            pair.setSecond(0);
                            if (allAssignments.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])) {
                                pair.setSecond(1);
                            }
                            return pair;
                        });
                        return pairs;
                    });
            calibrations.put(cardinality, new IsotonicRegression(stream));
        }



//        System.out.println("calibrating with isotonic regression");
        this.boosting = boosting;

    }


    public static CardinalityCalibrator train(StreamGenerator streamGenerator){
        Stream<Pair<Vector,Integer>> stream = streamGenerator.generateStream();
        ConcurrentHashMap<Integer, Integer> map = new ConcurrentHashMap<>();
        // vec(1) is cardinality
        stream.forEach(pair->map.put((int)pair.getFirst().get(1),1));
        Enumeration<Integer> ints = map.keys();
        List<Integer> cardinalities = new ArrayList<>();
        CardinalityCalibrator cardinalityCalibrator = new CardinalityCalibrator();
        cardinalityCalibrator.calibrations = new HashMap<>();
        while(ints.hasMoreElements()){
            cardinalities.add(ints.nextElement());
        }
        Collections.sort(cardinalities);

        for (int cardinality: cardinalities){
            Stream<Pair<Vector,Integer>> filtered = streamGenerator.generateStream().filter(pair->(int)pair.getFirst().get(1)==cardinality);
            cardinalityCalibrator.calibrations.put(cardinality,IsotonicRegression.train(filtered));
        }
        return cardinalityCalibrator;
    }



    public double calibrate(double uncalibrated, int cardinality){
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


    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double uncalibrated = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
        int cardinality = multiLabel.getNumMatchedLabels();
        return calibrations.get(cardinality).predict(uncalibrated);
    }


    @Override
    /**
     * assuming the first dim is uncalibrated score and the second dim is cardinality
     */
    public double predict(Vector vector) {
        return calibrate(vector.get(0), (int)vector.get(1));
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }
}
