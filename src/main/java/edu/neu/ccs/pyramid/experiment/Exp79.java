package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.TopFeatures;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * logistic regression analysis
 * can take any logistic regression(ridge, lasso, elasticnet)
 * produced by other exps
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

        if (config.getBoolean("verify.topFeatures")) {
            int limit = config.getInt("verify.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0, logisticRegression.getNumClasses())
                    .mapToObj(k -> LogisticRegressionInspector.topFeatures(logisticRegression, k, limit))
                    .collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = config.getString("verify.topFeatures.file");
            mapper.writeValue(new File(config.getString("output.folder"), file), topFeaturesList);

        }


        if (config.getBoolean("verify.topSeeds")) {
            int limit = config.getInt("verify.topSeeds.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0, logisticRegression.getNumClasses())
                    .mapToObj(k -> LogisticRegressionInspector.topFeatures(logisticRegression, k, logisticRegression.getNumFeatures()))
                    .collect(Collectors.toList());

            String file = config.getString("verify.topSeeds.file");

            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(config.getString("output.folder"), file)));
            for (int k=0;k<logisticRegression.getNumClasses();k++){
                TopFeatures topFeatures = topFeaturesList.get(k);
                List<String> topUnigrams = topFeatures.getTopFeatures().stream().filter(feature -> feature instanceof Ngram)
                        .filter(feature -> ((Ngram)feature).getN()==1)
                        .map(feature -> ((Ngram)feature).getNgram())
                        .limit(limit)
                        .collect(Collectors.toList());
                for (String unigram: topUnigrams){
                    bufferedWriter.write(unigram);
                    bufferedWriter.newLine();
                }
            }

            bufferedWriter.close();
        }


        if (config.getBoolean("verify.compareFeatures")) {
            int limit = config.getInt("verify.compareFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0, logisticRegression.getNumClasses())
                    .mapToObj(k -> LogisticRegressionInspector.topFeatures(logisticRegression, k, limit))
                    .collect(Collectors.toList());

            List<List<Ngram>> ngrams = topFeaturesList.stream().map(topfeatures -> topfeatures.getTopFeatures())
                    .map(featureList -> featureList.stream().filter(feature -> feature instanceof Ngram).map(feature -> (Ngram)feature)
                    .collect(Collectors.toList())).collect(Collectors.toList());

            String file = config.getString("verify.compareFeatures.file");

            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(config.getString("output.folder"), file)));

            List<FeatureComparision> comparisions = new ArrayList<>();
            for (int k=0;k<logisticRegression.getNumClasses();k++){
                List<Ngram> ngramList = ngrams.get(k);
                for (Ngram ngram: ngramList){
                    List<FeatureComparision> comparisionsForThis = new ArrayList<>();
                    for (int j=k+1;j<logisticRegression.getNumClasses();j++){
                        List<Ngram> against = ngrams.get(j);
                        for (Ngram ngram1:against){
                            if(Ngram.overlap(ngram,ngram1)){
                                FeatureComparision comparision = new FeatureComparision();
                                comparision.add(ngram,k);
                                comparision.add(ngram1,j);
                                comparisionsForThis.add(comparision);
                            }
                        }
                    }
                    if (comparisionsForThis.size()<=100){
                        comparisions.addAll(comparisionsForThis);
                    }
                }

            }

            for (int i=0;i<comparisions.size();i++){
                bufferedWriter.write(comparisions.get(i).toString());
                bufferedWriter.newLine();
            }

            bufferedWriter.close();
        }



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

    public static class FeatureComparision{
        private List<Ngram> features;
        private List<Integer> labels;

        public FeatureComparision() {
            features = new ArrayList<>();
            labels = new ArrayList<>();
        }

        public void add(Ngram feature, int label){
            features.add(feature);
            labels.add(label);
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            for (int i=0;i<features.size();i++){
                sb.append(features.get(i).getNgram());
                sb.append(":");
                sb.append(labels.get(i));
                sb.append("\t");
            }
            return sb.toString();
        }
    }

}
