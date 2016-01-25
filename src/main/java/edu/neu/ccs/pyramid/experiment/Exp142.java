package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.List;
import java.util.stream.IntStream;

/**
 * analyze "reflection"
 * Created by chengli on 1/23/16.
 */
public class Exp142 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "nuswide-128/data_sets/test"),
                DataSetType.ML_CLF_DENSE, true);
        long c= IntStream.range(0,testSet.getNumDataPoints()).filter(i-> testSet.getMultiLabels()[i].matchClass(49))
                .count();
        System.out.println(c);

        BMMClassifier bmmClassifier = (BMMClassifier) Serialization
                .deserialize("/Users/chengli/Documents/mixture_analysis/model_mix_lr_30");

        List<MultiLabel> samples = bmmClassifier.samples(testSet.getRow(74686));
        System.out.println(samples);
        System.out.println("marginal for lake = "+marginal(samples,34));
        System.out.println("marginal for water = "+marginal(samples,75));
        System.out.println("marginal for sunset = "+marginal(samples,62));
        System.out.println("marginal for reflection = "+marginal(samples,49));
    }


    static double marginal(List<MultiLabel> samples, int label){
        return IntStream.range(0,samples.size()).filter(i-> samples.get(i).matchClass(label)).count()/((double)samples.size());
    }
}
