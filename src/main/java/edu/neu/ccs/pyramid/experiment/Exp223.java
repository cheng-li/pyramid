package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * check whether certain combination occurs in train
 * Created by chengli on 1/17/16.
 */
public class Exp223 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{

        MultiLabelClfDataSet trainset = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "meka_imdb/1/data_sets/train"),
                DataSetType.ML_CLF_SPARSE, true);

        int[] set = {2,10};
        List<String> arr = IntStream.range(0,trainset.getNumDataPoints()).filter(i -> containAll(set, trainset.getMultiLabels()[i]))
                .mapToObj(i-> trainset.getIdTranslator().toExtId(i))
                .collect(Collectors.toList());
        System.out.println(arr);

    }

    private static boolean containAll(int[] set, MultiLabel multiLabel){
        boolean r = true;
        for (int i: set){
            if (!multiLabel.matchClass(i)){
                r = false;
            }
        }
        return r;
    }
}
