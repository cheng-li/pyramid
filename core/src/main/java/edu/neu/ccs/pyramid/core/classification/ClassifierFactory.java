package edu.neu.ccs.pyramid.core.classification;

import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;

import java.io.File;
import java.io.Serializable;

/**
 * Created by chengli on 10/4/14.
 */
public interface ClassifierFactory extends Serializable{
    Classifier train(ClfDataSet dataSet, TrainConfig config);
    Classifier deserialize(File file) throws Exception;
    default Classifier deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }
}
