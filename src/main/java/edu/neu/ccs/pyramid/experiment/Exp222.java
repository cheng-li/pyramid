package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * local version of exp221
 * Created by chengli on 1/16/16.
 */
public class Exp222 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{

        List<String> name = names();
        Map<String,String> map = mapToUrl();

        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "nuswide-128/data_sets//test"),
                DataSetType.ML_CLF_DENSE, true);
        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize("/Users/chengli/Documents/mixture_analysis/model_mix_lr");


        BMMClassifier independent = (BMMClassifier)Serialization.deserialize("/Users/chengli/Documents/mixture_analysis/model_independent_lr");

        System.out.println("classifier loaded");

        MultiLabel[] indePredictions = (MultiLabel[])Serialization.deserialize("/Users/chengli/Documents/mixture_analysis/prediction_independent_lr");
        MultiLabel[] bmmPredictions = (MultiLabel[])Serialization.deserialize("/Users/chengli/Documents/mixture_analysis/prediction_mix_lr");
        MultiLabel[] expectationPredictions = (MultiLabel[])Serialization.deserialize("/Users/chengli/Documents/mixture_analysis/prediction_mix_lr_expectation");

//        MultiLabel[] indePredictions = independent.predict(testSet);
//        MultiLabel[] bmmPredictions = bmmClassifier.predict(testSet);
//        MultiLabel[] expectationPredictions = new MultiLabel[testSet.getNumDataPoints()];
//        for (int i=0;i<testSet.getNumDataPoints();i++){
//            expectationPredictions[i] = bmmClassifier.predictByExpectation(testSet.getRow(i));
//        }
//
//        Serialization.serialize(indePredictions,new File(TMP,"prediction_independent_lr"));
//        Serialization.serialize(bmmPredictions,new File(TMP,"prediction_mix_lr"));
//        Serialization.serialize(expectationPredictions,new File(TMP,"prediction_mix_lr_expectation"));

        System.out.println("prediction done");

        boolean measure = true;
        if (measure){


            System.out.println("independent measures");

            LabelBasedMeasures indeMeasure = new LabelBasedMeasures(testSet,indePredictions);
            System.out.println(indeMeasure);
            MacroMeasures indemacro = new MacroMeasures(testSet,indePredictions);
            System.out.println(indemacro);
            System.out.println("acc = "+ Accuracy.accuracy(testSet.getMultiLabels(),indePredictions));
            System.out.printf("overlap = "+ Overlap.overlap(testSet.getMultiLabels(),indePredictions));

            System.out.println("bmm measures");

            LabelBasedMeasures bmmMeasures = new LabelBasedMeasures(testSet,bmmPredictions);
            System.out.println(bmmMeasures);
            MacroMeasures bmmmacro = new MacroMeasures(testSet,bmmPredictions);
            System.out.println(bmmmacro);
            System.out.println("acc = "+ Accuracy.accuracy(testSet.getMultiLabels(),bmmPredictions));
            System.out.println("overlap = " + Overlap.overlap(testSet.getMultiLabels(), bmmPredictions));
        }






        IdTranslator idTranslator = testSet.getIdTranslator();
        LabelTranslator labelTranslator = testSet.getLabelTranslator();

        List<String> extLabels = new ArrayList<>();
        for (int l=0;l<testSet.getNumClasses();l++){
            extLabels.add("\""+labelTranslator.toExtLabel(l)+"\"");
        }
        System.out.println(extLabels);

        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(testSet.getRow(i));
            double perplexity = Math.pow(2, Entropy.entropy2Based(proportions));
            MultiLabel pred = bmmPredictions[i];
            MultiLabel expectation =expectationPredictions[i];
            MultiLabel indePred = indePredictions[i];

            boolean condition = trueLabel.getMatchedLabels().size()>=2&&perplexity>1.5&&pred.equals(trueLabel)&&!expectation.equals(trueLabel)&&!indePred.equals(trueLabel);
//            boolean condition = trueLabel.getMatchedLabels().size()>=2&&perplexity>1.5&&pred.equals(trueLabel)&&!indePred.equals(trueLabel)&&!expectation.equals(trueLabel);
//            boolean condition = i==526;
//            boolean condition = !trueLabel.equals(pred);

            if (condition){
                System.out.println("----------------------------------------------");
                System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
                System.out.println(map.get(name.get(i)));
                System.out.println("labels = "+trueLabel.toStringWithExtLabels(labelTranslator));
                System.out.println("independent prediction = "+indePred.toStringWithExtLabels(labelTranslator));
                List<String> predString = new ArrayList<>();
                for (int l: pred.getMatchedLabels()){
                    predString.add("\""+labelTranslator.toExtLabel(l)+"\"");
                }
                System.out.println("mixture prediction = "+predString);
                System.out.println("expectation prediction= "+expectation.toStringWithExtLabels(labelTranslator));
                System.out.println("bmm probability for independent prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),indePred));
                System.out.println("bmm probability for mixture prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),pred));
                System.out.println("bmm probability for expectation prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),expectation));
                BMMInspector.covariance(bmmClassifier,testSet.getRow(i),labelTranslator);
                BMMInspector.visualizePrediction(bmmClassifier,testSet.getRow(i),labelTranslator);

//                for (int label: indePredictions[i].getMatchedLabels()){
//                    LogisticRegression logisticRegression = (LogisticRegression)independent.getBinaryClassifiers()[0][label];
//                    logisticRegression.setFeatureList(testSet.getFeatureList());
//                    ClassScoreCalculation classScoreCalculation = LogisticRegressionInspector.decisionProcess(logisticRegression, testSet.getLabelTranslator(), testSet.getRow(i), 1, 10);
//                    System.out.println("for class "+labelTranslator.toExtLabel(label));
//                    System.out.println(classScoreCalculation);
//                }
            }
        }
    }

    static List<String> names() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Documents/mixture_analysis/TestImagelist.txt"));
        List<String> list= new ArrayList<>();
        for (String line : lines){
            list.add(line);
//            System.out.println(sub);
        }

        return lines;
    }

    static Map<String, String> mapToUrl() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Documents/mixture_analysis/NUS-WIDE-urls.txt"));
        Map<String,String> map = new HashMap<>();
        for (String line : lines){
            String[] split = line.split("\\s+");
            String first = split[0];

            map.put(first,split[2]);
        }
//        System.out.println(map.get("0428_242362437.jpg"));
        return map;
    }
}
