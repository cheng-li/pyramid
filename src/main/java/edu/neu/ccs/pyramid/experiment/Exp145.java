package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Serialization;

/**
 * check bmm model
 * Created by chengli on 1/30/16.
 */
public class Exp145 {
    public static void main(String[] args) throws Exception{
        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize("/Users/chengli/tmp/model/c10.1.1.0.6.model/model");
        System.out.println(bmmClassifier);
    }

}
