package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;

import java.util.stream.IntStream;

/**
 * Created by chengli on 8/13/14.
 */
public interface Classifier {
    int predict(FeatureRow featureRow);

    default int[] predict(ClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                map(i -> predict(dataSet.getFeatureRow(i))).toArray();
    }

}
