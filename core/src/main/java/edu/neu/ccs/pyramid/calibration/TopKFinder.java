package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;


/**
 * find top K sets by calibrated probability
 */
public class TopKFinder {

    public static List<Pair<MultiLabel,Double>> topK(Vector x, CBM cbm, LabelCalibrator labelCalibrator,
                                                     VectorCalibrator vectorCalibrator, PredictionVectorizer predictionVectorizer,
                                                     int top){
        List<Pair<MultiLabel,Double>> list = new ArrayList<>();
        double[] marginals = labelCalibrator.calibratedClassProbs(cbm.predictClassProbs(x));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        BMDistribution bmDistribution = cbm.computeBM(x,0.001);
        for (int i=0;i<top;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            Map<MultiLabel,Integer> positionMap = new HashMap<>();
            positionMap.put(candidate,i);
            Vector yFeature = predictionVectorizer.feature(bmDistribution,candidate,marginals, Optional.of(positionMap));
            double pro = vectorCalibrator.calibrate(yFeature);
            list.add(new Pair<>(candidate,pro));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }

    public static List<Pair<MultiLabel,Double>> topKinSupport(Vector x, CBM cbm, LabelCalibrator labelCalibrator,
                                                     VectorCalibrator vectorCalibrator, PredictionVectorizer predictionVectorizer,
                                                     List<MultiLabel> support,
                                                     int top){
        Set<MultiLabel> supportSets = new HashSet<>(support);
        List<Pair<MultiLabel,Double>> list = new ArrayList<>();
        double[] marginals = labelCalibrator.calibratedClassProbs(cbm.predictClassProbs(x));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        BMDistribution bmDistribution = cbm.computeBM(x,0.001);
        int count = 0;
        for (int i=0;i<Integer.MAX_VALUE;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (supportSets.contains(candidate)) {
                Map<MultiLabel,Integer> positionMap = new HashMap<>();
                positionMap.put(candidate,i);
                Vector yFeature = predictionVectorizer.feature(bmDistribution,candidate,marginals, Optional.of(positionMap));
                double pro = vectorCalibrator.calibrate(yFeature);
                list.add(new Pair<>(candidate,pro));
                count += 1;
            }
            if (count==top){
                break;
            }
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }


}
