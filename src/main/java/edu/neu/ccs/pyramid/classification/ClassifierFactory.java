package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.io.File;
import java.io.IOException;

/**
 * Created by chengli on 10/4/14.
 */
public interface ClassifierFactory {
    Classifier train(ClfDataSet dataSet, TrainConfig config);
    Classifier deserialize(File file) throws Exception;
    default Classifier deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }
}
