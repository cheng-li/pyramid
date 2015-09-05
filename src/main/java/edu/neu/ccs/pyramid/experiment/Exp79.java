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
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.SpanNotNgram;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
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
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        (new File(config.getString("output.folder"))).mkdirs();

//        showSkip(config);

        if (config.getBoolean("verify")){
            verify(config);
        }


        if (config.getBoolean("test")){
            test(config);
        }

        if (config.getBoolean("count")){
            count(config);
        }


        if (config.getBoolean("weight")){
            weight(config);
        }
    }

    public static void verify(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);

        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));

        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));

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
                    for (int j=0;j<logisticRegression.getNumClasses();j++){
                        if (j!=k){
                            List<Ngram> against = ngrams.get(j);
                            for (Ngram ngram1:against){
                                boolean condition = ngram.getN()==1&&ngram1.getN()==2
                                        &&Ngram.overlap(ngram,ngram1)&&ngram.getSlop()==0&&ngram1.getSlop()==0;
                                if(condition){
                                    FeatureComparision comparision = new FeatureComparision();
                                    comparision.add(ngram,k);
                                    comparision.add(ngram1,j);
                                    comparisionsForThis.add(comparision);
                                }
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


            List<SpanNotNgram> spanNotNgrams = new ArrayList<>();
            for (FeatureComparision comparision: comparisions){
                Ngram ngram1 = comparision.features.get(0);
                Ngram ngram2 = comparision.features.get(1);
                SpanNotNgram spanNotNgram = new SpanNotNgram();
                spanNotNgram.setInclude(ngram1);
                spanNotNgram.setExclude(ngram2);
                spanNotNgrams.add(spanNotNgram);
            }

            Serialization.serialize(spanNotNgrams,new File(config.getString("output.folder"), "spanNotNgrams.ser"));
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

    public static void count(Config config) throws Exception{
        System.out.println("count");
        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));

        Set<Integer> selected = LogisticRegressionInspector.usedFeaturesCombined(logisticRegression);

        List<Ngram> ngrams = logisticRegression.getFeatureList().getAll().stream()
                .filter(feature -> selected.contains(feature.getIndex()))
                .filter(feature -> feature instanceof Ngram)
                .map(feature -> (Ngram) feature).collect(Collectors.toList());

        int maxN = ngrams.stream().mapToInt(Ngram::getN).max().getAsInt();
        int maxSlop = ngrams.stream().mapToInt(Ngram::getSlop).max().getAsInt();
        double[][] counts = new double[maxN][maxSlop+1];
        ngrams.stream().forEach(ngram ->{
            int n = ngram.getN();
            int slop = ngram.getSlop();
            counts[n-1][slop] += 1;
        });

        double total = ngrams.size();
        System.out.println("total = "+total);
        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                counts[i][j] /= total;
            }
        }

        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                System.out.println("n="+(i+1)+", slop="+j+", p="+counts[i][j]);
            }
        }


    }


    public static void weight(Config config) throws Exception{
        System.out.println("weight");
        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));

        List<Ngram> ngrams = logisticRegression.getFeatureList().getAll().stream()
                .map(feature-> (Ngram)feature).collect(Collectors.toList());

        int maxN = ngrams.stream().mapToInt(Ngram::getN).max().getAsInt();
        int maxSlop = ngrams.stream().mapToInt(Ngram::getSlop).max().getAsInt();
        double[][] counts = new double[maxN][maxSlop+1];


        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double weight = element.get();
                Ngram ngram = ngrams.get(featureIndex);
                int n = ngram.getN();
                int slop = ngram.getSlop();
                counts[n-1][slop] += Math.abs(weight);
            }
        }


        double total = 0;

        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                total+= counts[i][j];
            }
        }

        System.out.println("total = "+total);
        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                counts[i][j] /= total;
            }
        }

        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                System.out.println("n="+(i+1)+", slop="+j+", p="+counts[i][j]);
            }
        }


    }

    public static void showSkip(Config config) throws Exception{
        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));

        List<TopFeatures> topFeaturesList = IntStream.range(0,2)
                .mapToObj(k -> LogisticRegressionInspector.topFeatures(logisticRegression,k,1000))
                .collect(Collectors.toList());
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(config.getString("output.folder"),"top.json"), topFeaturesList);


    }


    public static void test(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "test.trec"),
                DataSetType.CLF_SPARSE, true);

        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(config.getString("input.model")));

        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));

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
