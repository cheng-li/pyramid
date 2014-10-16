package edu.neu.ccs.pyramid.regression.m_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureType;


import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;

/**
 * Created by chengli on 8/6/14.
 */
class Splitter {
    /**
     *
     * @param regTreeConfig
     * @param dataAppearance
     * @return best valid splitResult, possibly nothing
     */
    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       int[] dataAppearance){
        int[] activeFeatures = regTreeConfig.getActiveFeatures();
        return Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        dataAppearance,featureIndex))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .max(Comparator.comparing(SplitResult::getReduction));
    }

    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       int[] dataAppearance,
                                       int featureIndex){
        Optional<SplitResult> splitResult;
        FeatureType featureType = dataSet
                .getFeatureColumn(featureIndex).getSetting()
                .getFeatureType();
        if (featureType==FeatureType.NUMERICAL){
            splitResult = IntervalSplitter.split(regTreeConfig,dataSet,labels,
                    dataAppearance,featureIndex);
        } else if(featureType==FeatureType.BINARY){
            splitResult = BinarySplitter.split(regTreeConfig,dataSet,labels,
                    dataAppearance,featureIndex);
        } else{
            throw new IllegalArgumentException("unsupported feature type");
        }
        return splitResult;
    }
}
