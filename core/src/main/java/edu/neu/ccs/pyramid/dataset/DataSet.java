package edu.neu.ccs.pyramid.dataset;


import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

/**
 * Created by chengli on 8/4/14.
 */
public interface DataSet extends Serializable{
    int getNumDataPoints();
    int getNumFeatures();
    Vector getColumn(int featureIndex);
    Vector getRow(int dataPointIndex);
    void setFeatureValue(int dataPointIndex,
                                int featureIndex, double featureValue);
    boolean isDense();
    boolean hasMissingValue();
    String getMetaInfo();

    IdTranslator getIdTranslator();
    FeatureList getFeatureList();
    void setFeatureList(FeatureList featureList);
    void setIdTranslator(IdTranslator idTranslator);

    Density density();

}
