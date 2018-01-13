package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.AveragePrecision;
import edu.neu.ccs.pyramid.eval.MAP;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.AccPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.MarginalPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.PluginF1;
import edu.neu.ccs.pyramid.util.FileUtil;
import edu.neu.ccs.pyramid.util.Serialization;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CBMEval {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        CBM cbm = (CBM) Serialization.deserialize(config.getString("model"));
        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(config.getString("support"));
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("data"), DataSetType.ML_CLF_SEQ_SPARSE,true);
        List<Integer> skipped = config.getIntegers("skippedLabels");

        Set<Integer> skipSet = new HashSet<>(skipped);
        List<Integer> keep = IntStream.range(0,test.getNumClasses())
                .filter(a->!skipSet.contains(a))
                .boxed().collect(Collectors.toList());
        System.out.println("label averaged MAP");
        System.out.println(MAP.map(cbm, test, keep));
        System.out.println("instance averaged MAP");
        System.out.println(MAP.instanceMAP(cbm, test));
        System.out.println("global AP truncated at 30");
        System.out.println(AveragePrecision.globalAveragePrecisionTruncated(cbm, test, 30));


        reportAccPrediction(cbm, test);
        reportF1Prediction(cbm, support, test);
        reportHammingPrediction(cbm, test);
    }



    private static void reportAccPrediction(CBM cbm, MultiLabelClfDataSet dataSet) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions with the instance set accuracy optimal predictor");
        AccPredictor accPredictor = new AccPredictor(cbm);
        accPredictor.setComponentContributionThreshold(0.001);
        MultiLabel[] predictions = accPredictor.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
        System.out.println(mlMeasures);
    }


    private static void reportF1Prediction(CBM cbm, List<MultiLabel> support, MultiLabelClfDataSet dataSet) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions with the instance F1 optimal predictor");
        PluginF1 pluginF1 = new PluginF1(cbm);
        pluginF1.setSupport(support);
        pluginF1.setPiThreshold(0.001);
        MultiLabel[] predictions = pluginF1.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
        System.out.println(mlMeasures);
    }

    private static void reportHammingPrediction(CBM cbm, MultiLabelClfDataSet dataSet) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions with the instance Hamming loss optimal predictor");
        MarginalPredictor marginalPredictor = new MarginalPredictor(cbm);
        marginalPredictor.setPiThreshold(0.001);
        MultiLabel[] predictions = marginalPredictor.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);

        System.out.println(mlMeasures);
    }
}
