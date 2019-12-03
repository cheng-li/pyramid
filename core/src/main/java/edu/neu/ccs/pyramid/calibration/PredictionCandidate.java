package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.List;

public class PredictionCandidate {
    public Vector x;
    public MultiLabel multiLabel;
    public double[] labelProbs;
    public List<Pair<MultiLabel,Double>> sparseJoint;


}
