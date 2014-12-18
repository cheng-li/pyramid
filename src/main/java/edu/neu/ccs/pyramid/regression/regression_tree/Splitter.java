package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureType;


import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Created by chengli on 8/6/14.
 */
public class Splitter {
    /**
     *
     * @param regTreeConfig
     * @param probs
     * @return best valid splitResult, possibly nothing
     */
    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs){
        int[] activeFeatures = regTreeConfig.getActiveFeatures();
        return Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        probs,featureIndex))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .max(Comparator.comparing(SplitResult::getReduction));
    }


    public static List<SplitResult> getAllSplits(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs){
        int[] activeFeatures = regTreeConfig.getActiveFeatures();
        return Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        probs,featureIndex))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());
    }

    public static List<SplitResult> getAllSplits(RegTreeConfig regTreeConfig,
                                                 DataSet dataSet,
                                                 double[] labels
                                                 ){
        double[] probs = new double[labels.length];
        for (int i=0;i<labels.length;i++){
            probs[i] = 1;
        }
        return getAllSplits(regTreeConfig,dataSet,labels,probs);
    }

    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs,
                                       int featureIndex){
        Optional<SplitResult> splitResult;
        FeatureType featureType = dataSet
                .getFeatureSetting(featureIndex)
                .getFeatureType();
        if (featureType==FeatureType.NUMERICAL){
            splitResult = IntervalSplitter.split(regTreeConfig,dataSet,labels,
                    probs,featureIndex);
        } else if(featureType==FeatureType.BINARY){
            splitResult = BinarySplitter.split(regTreeConfig,dataSet,labels,
                    probs,featureIndex);
        } else{
            throw new IllegalArgumentException("unsupported feature type");
        }
        return splitResult;
    }
}
