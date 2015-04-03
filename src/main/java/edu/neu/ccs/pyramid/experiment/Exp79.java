package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * logistic regression analysis
 * Created by chengli on 3/31/15.
 */
public class Exp79 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        (new File(config.getString("output.folder"))).mkdirs();

        if (config.getBoolean("verify")){
            verify(config);
        }


        if (config.getBoolean("test")){
            test(config);
        }
    }

    public static void verify(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);

        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));
        List<Integer> candidates = new ArrayList<>();
        int[] prediction = logisticRegression.predict(dataSet);
        int[] labels = dataSet.getLabels();
        for (int i=0;i<prediction.length;i++){
            if (labels[i]==prediction[i] && config.getBoolean("verify.analyze.doc.withRightPrediction")){
                candidates.add(i);
            }
            if (labels[i]!=prediction[i] && config.getBoolean("verify.analyze.doc.withWrongPrediction")){
                candidates.add(i);
            }
        }
        int limit = config.getInt("verify.analyze.rule.limit");
        List<PredictionAnalysis> predictionAnalysisList = candidates.parallelStream()
                .map(i -> LogisticRegressionInspector.analyzePrediction(logisticRegression, dataSet, i, limit))
                .collect(Collectors.toList());
        ObjectMapper mapper = new ObjectMapper();
        String file = config.getString("verify.analyze.file");
        mapper.writeValue(new File(config.getString("output.folder"),file), predictionAnalysisList);
    }


    public static void test(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "test.trec"),
                DataSetType.CLF_SPARSE, true);

        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));
        List<Integer> candidates = new ArrayList<>();
        int[] prediction = logisticRegression.predict(dataSet);
        int[] labels = dataSet.getLabels();
        for (int i=0;i<prediction.length;i++){
            if (labels[i]==prediction[i] && config.getBoolean("test.analyze.doc.withRightPrediction")){
                candidates.add(i);
            }
            if (labels[i]!=prediction[i] && config.getBoolean("test.analyze.doc.withWrongPrediction")){
                candidates.add(i);
            }
        }
        int limit = config.getInt("test.analyze.rule.limit");
        List<PredictionAnalysis> predictionAnalysisList = candidates.parallelStream()
                .map(i -> LogisticRegressionInspector.analyzePrediction(logisticRegression, dataSet, i, limit))
                .collect(Collectors.toList());
        ObjectMapper mapper = new ObjectMapper();
        String file = config.getString("test.analyze.file");
        mapper.writeValue(new File(config.getString("output.folder"),file), predictionAnalysisList);
    }
}
