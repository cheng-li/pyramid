package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * cluster vs performance plot
 * Created by chengli on 1/26/16.
 */
public class Exp144 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        String folder = "/Users/chengli/Documents/mixture_analysis/cluster_models/";
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "nuswide-128/data_sets/test"),
                DataSetType.ML_CLF_DENSE, true);
        int[] clusters = {1,5,10,15,20,25,30,35,40};
        List<Double> accs = new ArrayList<>();
        List<Double> overlaps = new ArrayList<>();
        for (int k:clusters){
            System.out.println(k);
            BMMClassifier bmmClassifier = (BMMClassifier) Serialization
                    .deserialize(folder+k);
            MultiLabel[] pred = bmmClassifier.predict(testSet);
            accs.add(Accuracy.accuracy(testSet.getMultiLabels(), pred));
            overlaps.add( Overlap.overlap(testSet.getMultiLabels(), pred));
        }
        System.out.println(Arrays.toString(clusters));
        System.out.println(accs);
        System.out.println(overlaps);
    }
}
