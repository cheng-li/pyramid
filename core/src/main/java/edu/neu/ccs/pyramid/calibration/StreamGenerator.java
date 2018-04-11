package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.stream.Stream;

public class StreamGenerator {

    private Config config;
    private IMLGradientBoosting boosting;
    private MultiLabelClfDataSet dataSet;

    public  Stream<Pair<Vector,Integer>> generateStream(){
        return null;
    }
}
