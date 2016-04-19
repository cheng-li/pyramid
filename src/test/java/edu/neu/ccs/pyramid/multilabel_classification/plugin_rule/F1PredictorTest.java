package edu.neu.ccs.pyramid.multilabel_classification.plugin_rule;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.neu.ccs.pyramid.multilabel_classification.Enumerator;
import edu.neu.ccs.pyramid.multilabel_classification.LossMatrixGenerator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by chengli on 4/5/16.
 */
public class F1PredictorTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }


    private static void test1(){
        List<MultiLabel> multiLabelList = Enumerator.enumerate(2);
        List<Double> probs = new ArrayList<>();
        probs.add(0.1);
        probs.add(0.2);
        probs.add(0.3);
        probs.add(0.4);
//        Matrix p = F1Predictor.getPMatrix(2,multiLabelList,probs);
//        Matrix delta = F1Predictor.getDeltaMatrix(p);
        System.out.println(F1Predictor.predict(2,multiLabelList,probs));

    }

    private static void test2(){
        int numLabels = 3;
        Matrix matrix = LossMatrixGenerator.matrix(numLabels,"f1");
        List<MultiLabel> multiLabels = Enumerator.enumerate(numLabels);
        List<Double> dis = LossMatrixGenerator.sampleDistribution(numLabels);
        MultiLabel pred = F1Predictor.predict(numLabels,multiLabels,dis);
        MultiLabel search = F1Predictor.exhaustiveSearch(numLabels,matrix,dis);
        System.out.println("pred = "+pred);
        System.out.println("search = "+search);



    }


//    private static void test3() throws Exception{
////        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
////                DataSetType.ML_CLF_DENSE, true);
////        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
////                DataSetType.ML_CLF_DENSE, true);
//
//        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "nuswide-128/data_sets/train"),
//                DataSetType.ML_CLF_DENSE, true);
//        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "nuswide-128/data_sets/test"),
//                DataSetType.ML_CLF_DENSE, true);
//        CBMClassifier cbm = (CBMClassifier) Serialization.deserialize(new File(TMP,"nuswide_model"));
//
//        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
//        PluginF1 pluginF1 = new PluginF1(cbm, support);
//        pluginF1.setPredictionMode("sampling");
//        pluginF1.setNumSamples(100);
//
//        System.out.println("training set performance");
//        System.out.println(new MLMeasures(pluginF1,trainSet));
//
//
//        System.out.println("test set performance");
//        System.out.println(new MLMeasures(pluginF1,testSet));
//    }

}