package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.SplitResult;
import edu.neu.ccs.pyramid.regression.regression_tree.Splitter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 12/13/14.
 */
public class SeedExtractor {
    private List<List<String>> seedsForAllClasses;

    public SeedExtractor(ClfDataSet dataSet){
        seedsForAllClasses = new ArrayList<>();
        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(dataSet.getNumClasses());
        priorProbClassifier.fit(dataSet);
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.setActiveFeatures(activeFeatures);
        Comparator<SplitResult> comparator = Comparator.comparing(SplitResult::getReduction);
        for (int k=0;k<dataSet.getNumClasses();k++){
            double[] gradient = priorProbClassifier.getGradient(dataSet,k);
            List<String> seeds = Splitter.getAllSplits(regTreeConfig, dataSet, gradient)
                    .stream().sorted(comparator.reversed()).map(SplitResult::getFeatureIndex).
                            map(i -> dataSet.getFeatureSetting(i).getFeatureName())
                    .collect(Collectors.toList());
            seedsForAllClasses.add(seeds);
        }
    }

    public List<String> getSeeds(int classIndex, int limit){
        return seedsForAllClasses.get(classIndex).stream().limit(limit).collect(Collectors.toList());
    }

}
