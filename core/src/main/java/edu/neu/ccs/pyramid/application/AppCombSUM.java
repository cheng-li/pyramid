package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTAT;
import edu.neu.ccs.pyramid.calibration.CTFT;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.eval.Recall;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.ReportUtils;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class AppCombSUM {

    public static void main(String[] args)throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        logger.info(config.toString());


        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        List<String> modelPaths = config.getStrings("modelPaths");
        String testFolder = config.getString("testFolder");
        String validFolder = config.getString("validFolder");
        String calibFolder = config.getString("calibFolder");
        int limit = config.getInt("kvalue");
        double targetValue = config.getDouble("threshold.targetValue");
        String ensembleName = config.getString("ensembleModelName");

        List<String> validDocIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",validFolder+"_reports","report.csv").toString());
        List<String> testDocIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",testFolder+"_reports","report.csv").toString());
        List<String> calibDocIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",calibFolder+"_reports","report.csv").toString());

        List<Map<String,Map<String,Pair<Double,Double>>>> validMapsLists = new ArrayList<>();
        List<Map<String,Map<String,Pair<Double,Double>>>> testMapsLists = new ArrayList<>();
        List<Map<String,Map<String,Pair<Double,Double>>>> calibMapsLists = new ArrayList<>();


        for(int i = 0; i < modelPaths.size(); i++){
            String validReportpath = Paths.get(modelPaths.get(i),"predictions",validFolder+"_reports","top_sets.csv").toString();
            String testReportPath = Paths.get(modelPaths.get(i),"predictions",testFolder+"_reports","top_sets.csv").toString();
            String calibReportPath = Paths.get(modelPaths.get(i),"predictions",calibFolder+"_reports","top_sets.csv").toString();
            validMapsLists.add(gettopKset(validReportpath,limit));
            testMapsLists.add(gettopKset(testReportPath,limit));
            calibMapsLists.add(gettopKset(calibReportPath,limit));
        }
        Map<String,Pair<String,Double>> validMap = getFinalPrediction(validMapsLists);
        Map<String,Pair<String,Double>> testMap = getFinalPrediction(testMapsLists);
        Map<String,Pair<String,Double>> calibMap = getFinalPrediction(calibMapsLists);

        List<Map<String,Pair<Double,Double>>> validConfideceRankLists = new ArrayList<>();
        List<Map<String,Pair<Double,Double>>> testConfideceRankLists = new ArrayList<>();

        for(int i = 0; i < modelPaths.size(); i++) {
            String validReportPath = Paths.get(modelPaths.get(i),"predictions",validFolder+"_reports","top_sets.csv").toString();
            String testReportPath = Paths.get(modelPaths.get(i),"predictions",testFolder+"_reports","top_sets.csv").toString();
            Map<String,Map<String,Pair<Double,Double>>> validAllKSetsMap = gettopKset(validReportPath,20);
            Map<String,Map<String,Pair<Double,Double>>> testAllKSetsMap = gettopKset(testReportPath,20);
            validConfideceRankLists.add(getConfidenceRank(validAllKSetsMap,validMap,validDocIds));
            testConfideceRankLists.add(getConfidenceRank(testAllKSetsMap,testMap,testDocIds));

        }



        String validSetPath0 = modelPaths.get(0).split("model_predictions")[0]+"data_sets/"+validFolder;
        String testSetPath0 = modelPaths.get(0).split("model_predictions")[0]+"data_sets/"+testFolder;
        String calibSetPath0 = modelPaths.get(0).split("model_predictions")[0]+"data_sets/"+calibFolder;
        MultiLabelClfDataSet validSet0 = TRECFormat.loadMultiLabelClfDataSet(validSetPath0, DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet testSet0 = TRECFormat.loadMultiLabelClfDataSet(testSetPath0, DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet calibSet0 = TRECFormat.loadMultiLabelClfDataSet(calibSetPath0, DataSetType.ML_CLF_SPARSE,true);

        LabelTranslator newLabelTranslatorValid = AppEnsemble.getLabelTranslatorEnsemble(config,validFolder);
        LabelTranslator newLabelTranslatorTest = AppEnsemble.getLabelTranslatorEnsemble(config,testFolder);
        LabelTranslator newLabelTranslatorCalib = AppEnsemble.getLabelTranslatorEnsemble(config,calibFolder);

        Map<String,String> groundTruthValid = ReportUtils.getIDGroundTruth(validSet0);
        Map<String,String> groundTruthTest = ReportUtils.getIDGroundTruth(testSet0);
        Map<String,String> groundTruthCalib = ReportUtils.getIDGroundTruth(calibSet0);

        IsotonicRegression isotonicRegression = trainIsoRegression(calibMap,calibDocIds,config,newLabelTranslatorCalib,groundTruthCalib);

        generateReport(validMap,validDocIds,config,validFolder,newLabelTranslatorValid,groundTruthValid,validConfideceRankLists,isotonicRegression);
        generateReport(testMap,testDocIds,config,testFolder,newLabelTranslatorTest,groundTruthTest,testConfideceRankLists,isotonicRegression);




        if(config.getBoolean("tuneThreshold")){

            logger.info("start tuning confidence threshold");
            Stream<Pair<Double,Double>> streamValid;
            double threshold = 1.1;
            if(config.getString("threshold.targetMetric").equals("accuracy")){

                streamValid = edu.neu.ccs.pyramid.util.ReportUtils.getConfidenceCorrectness(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",validFolder+"_reports","report.csv").toString()).stream();
                CTAT.Summary validSummary = CTAT.findThreshold(streamValid,targetValue);
                threshold = validSummary.getConfidenceThreshold();

            }

            if(config.getString("threshold.targetMetric").equals("f1")){
                streamValid =  edu.neu.ccs.pyramid.util.ReportUtils.getConfidenceF1(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",validFolder+"_reports","report.csv").toString()).stream();
                CTFT.Summary summary_valid = CTFT.findThreshold(streamValid,targetValue);
                threshold = summary_valid.getConfidenceThreshold();
            }
            FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"models",
                    "threshold",config.getString("threshold.name")).toFile(),""+threshold);

            double confidenceThresholdClipped = CTAT.clip(threshold,config.getDouble("threshold.lowerBound"),config.getDouble("threshold.upperBound"));

            FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"models",
                    "threshold",config.getString("threshold.name")+"_clipped").toFile(),""+confidenceThresholdClipped);

            logger.info("tuning threshold is done");

            List<Pair<Double,Double>> testStream;
            if(config.getString("threshold.targetMetric").equals("accuracy")){
                testStream = edu.neu.ccs.pyramid.util.ReportUtils.getConfidenceCorrectness(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",testFolder+"_reports","report.csv").toString());
                CTAT.Summary testSummary_unclipped = CTAT.applyThreshold(testStream.stream(),threshold);
                CTAT.Summary testSummary_clipped = CTAT.applyThreshold(testStream.stream(),confidenceThresholdClipped);
                logger.info("*****************");
                logger.info("autocoding performance with unclipped CTAT "+testSummary_unclipped.getConfidenceThreshold());
                logger.info("autocoding percentage = "+ testSummary_unclipped.getAutoCodingPercentage());
                logger.info("autocoding accuracy = "+ testSummary_unclipped.getAutoCodingAccuracy());
                logger.info("number of autocoded documents = "+ testSummary_unclipped.getNumAutoCoded());
                logger.info("number of correct autocoded documents = "+ testSummary_unclipped.getNumCorrectAutoCoded());

                logger.info("*****************");

                logger.info("autocoding performance with clipped CTAT "+testSummary_clipped.getConfidenceThreshold());
                logger.info("autocoding percentage = "+ testSummary_clipped.getAutoCodingPercentage());
                logger.info("autocoding accuracy = "+ testSummary_clipped.getAutoCodingAccuracy());
                logger.info("number of autocoded documents = "+ testSummary_clipped.getNumAutoCoded());
                logger.info("number of correct autocoded documents = "+ testSummary_clipped.getNumCorrectAutoCoded());

            }

            if(config.getString("threshold.targetMetric").equals("f1")){
                testStream = ReportUtils.getConfidenceF1(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",testFolder+"_reports","report.csv").toString());
                CTFT.Summary summary_test = CTFT.applyThreshold(testStream.stream(),threshold);
                CTFT.Summary summary_test_clipped = CTFT.applyThreshold(testStream.stream(),confidenceThresholdClipped);
                logger.info("*****************");
                logger.info("autocoding performance with unclipped CTFT "+summary_test.getConfidenceThreshold());
                logger.info("autocoding percentage = "+ summary_test.getAutoCodingPercentage());
                logger.info("autocoding accuracy = "+ summary_test.getAutoCodingAccuracy());
                logger.info("autocoding F1 = "+ summary_test.getAutoCodingF1());
                logger.info("number of autocoded documents = "+ summary_test.getNumAutoCoded());
                logger.info("number of correct autocoded documents = "+ summary_test.getNumCorrectAutoCoded());

                logger.info("*****************");

                logger.info("autocoding performance with clipped CTFT "+summary_test_clipped.getConfidenceThreshold());
                logger.info("autocoding percentage = "+ summary_test_clipped.getAutoCodingPercentage());
                logger.info("autocoding accuracy = "+ summary_test_clipped.getAutoCodingAccuracy());
                logger.info("autocoding F1 = "+ summary_test_clipped.getAutoCodingF1());
                logger.info("number of autocoded documents = "+ summary_test_clipped.getNumAutoCoded());
                logger.info("number of correct autocoded documents = "+ summary_test_clipped.getNumCorrectAutoCoded());



            }


        }






        if (fileHandler!=null){
            fileHandler.close();
        }





    }


    private static IsotonicRegression trainIsoRegression(Map<String,Pair<String,Double>> map,List<String> docIds,Config config,LabelTranslator labelTranslator,Map<String,String> groundTruth)throws Exception {

        double[] locations = new double[docIds.size()];
        double[] numbers = new double[docIds.size()];
        for (int i = 0; i < docIds.size(); i++) {
            String docId = docIds.get(i);
            Pair<String, Double> docInfo = map.get(docId);

            String[] labels = groundTruth.get(docId).split(",");

            MultiLabel pre = new MultiLabel();
            MultiLabel lab = new MultiLabel();

            String pres = docInfo.getFirst();
            if (!pres.isEmpty()) {
                String[] predictions = docInfo.getFirst().split(",");
                for (String prediction : predictions) {

                    pre.addLabel(labelTranslator.toIntLabel(prediction.trim()));
                }
            }

            for (String label : labels) {

                lab.addLabel(labelTranslator.toIntLabel(label.trim()));
            }


            double truth = 0;
            if (pre.getMatchedLabels().equals(lab.getMatchedLabels())) {
                truth = 1;
            }

            double confidence = map.get(docId).getSecond();
            locations[i] = confidence;
            numbers[i] = truth;

        }

        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers);
        Serialization.serialize(isotonicRegression, Paths.get(config.getString("output.folder"), "model_predictions", config.getString("ensembleModelName"), "models","set_calibrator"));
        return isotonicRegression;

    }

    private static void generateReport(Map<String,Pair<String,Double>> map,List<String> docIds,Config config,String dataSetFolder,LabelTranslator labelTranslator,Map<String,String> groundTruth,List<Map<String,Pair<Double,Double>>> confideceRankLists,IsotonicRegression isotonicRegression)throws Exception{
        StringBuilder sb = new StringBuilder();
        sb.append("doc_id").append("\t").append("prediction").append("\t").append("prediction_type").append("\t")
                .append("confidence").append("\t").append("truth").append("\t").append("ground_truth").append("\t")
                .append("precison").append("\t").append("recall").append("\t").append("F1").append("\t")
                .append("gb-tenant_confidence").append("\t").append("gb-tenant_rank").append("\t").append("lr-tenant_confidence").append("\t")
                .append("lr-tenant_rank").append("\t").append("gb-common_confidence").append("\t").append("gb-common_rank").append("\t")
                .append("lr-common_confidence").append("\t").append("lr-common_rank").append("\n");



        for (int i = 0; i < docIds.size(); i++) {
            String docId = docIds.get(i);
            Pair<String,Double> docInfo = map.get(docId);

            sb.append(docId).append("\t").append(docInfo.getFirst()).append("\t").append("set").append("\t")
                    .append(isotonicRegression.predict(map.get(docId).getSecond())).append("\t");
            String[] labels = groundTruth.get(docId).split(",");

            MultiLabel pre = new MultiLabel();
            MultiLabel lab = new MultiLabel();

            String pres = docInfo.getFirst();
            if(!pres.isEmpty()){
                String[] predictions = docInfo.getFirst().split(",");
                for(String prediction:predictions){

                    pre.addLabel(labelTranslator.toIntLabel(prediction.trim()));
                }
            }

            for(String label : labels){

                lab.addLabel(labelTranslator.toIntLabel(label.trim()));
            }

            double precision = Precision.precision(lab,pre);
            double recall = Recall.recall(lab,pre);
            double f1 = FMeasure.f1(precision,recall);
            double truth = 0;
            if(pre.getMatchedLabels().equals(lab.getMatchedLabels())){
                truth =1;
            }

            sb.append(truth).append("\t").append(groundTruth.get(docId)).append("\t").append(precision).append("\t")
                    .append(recall).append("\t").append(f1).append("\t");

            for(int j = 0; j < config.getStrings("modelPaths").size(); j++){

                sb.append(confideceRankLists.get(j).get(docId).getFirst()).append("\t").append(confideceRankLists.get(j).get(docId).getSecond()).append("\t");
            }

            sb.append("\n");
        }

        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),"model_predictions",config.getString("ensembleModelName"),"predictions",dataSetFolder+"_reports","report.csv").toFile(),sb.toString());

    }


    private static Map<String,Pair<Double,Double>> getConfidenceRank(Map<String,Map<String,Pair<Double,Double>>> allksetsMap,Map<String,Pair<String,Double>> predictionMap,List<String> docIds){
        Map<String,Pair<Double,Double>> resultMap = new HashMap<>();
        for(int i = 0; i < docIds.size(); i++){
            String docId = docIds.get(i);
            Map<String,Pair<Double,Double>> topKsets = allksetsMap.get(docId);
            String prediction = predictionMap.get(docId).getFirst();
            for(String key : topKsets.keySet()){
                if(Arrays.stream(key.split(",")).sorted().collect(Collectors.toSet()).equals(Arrays.stream(prediction.split(",")).sorted().collect(Collectors.toSet()))){
                    resultMap.put(docId,topKsets.get(key));
                }
            }
            if(!resultMap.containsKey(docId)){
                resultMap.put(docId,new Pair<>(0.0,0.0));
            }

        }

        return resultMap;



    }









    private static Map<String,Map<String,Pair<Double,Double>>> gettopKset(String reportPath,int limit)throws Exception{

        Map<String,Map<String,Pair<Double,Double>>> mapDocsSets = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(reportPath));
        lines.stream().forEach(line->{
            String[] lineInfo = line.split("\t");
            String docId = lineInfo[0];
            String pres = lineInfo[1];
            double confi = Double.parseDouble(lineInfo[2]);
            if(!mapDocsSets.containsKey(docId)){
                Map<String,Pair<Double,Double>> map = new HashMap<>();
                mapDocsSets.put(docId,map);

            }
            if(mapDocsSets.get(docId).size() < limit) {
                double rank = mapDocsSets.get(docId).size()+1;
                mapDocsSets.get(docId).put(pres, new Pair<>(confi,rank));
            }
        });

        return mapDocsSets;



    }


    private static Map<String,Pair<String,Double>> getFinalPrediction(List<Map<String,Map<String,Pair<Double,Double>>>> list){
        List<String> docIds = list.get(0).keySet().stream().sorted().collect(Collectors.toList());
        Map<String,Pair<String,Double>> map = new HashMap<>();

        docIds.stream().forEach(docId->{
            map.put(docId,getFinalPredictionDoc(docId,list));
        });

        return map;
    }




    private static Pair<String,Double> getFinalPredictionDoc(String docId,List<Map<String,Map<String,Pair<Double,Double>>>> list){

        Set<String> set = new HashSet<>();

        for(int i = 0; i < list.size(); i++){
            if (list.get(i).get(docId)==null){
                System.out.println("list "+i +" does not contain "+docId);
            }
            set.addAll(list.get(i).get(docId).entrySet().stream().map(entry->entry.getKey()).collect(Collectors.toSet()));
        }

        Comparator<Pair<String,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<String,Double>> listResult = new ArrayList<>();
        for(String codeSet : set){

            double confidenceSum = 0;

            for(int i = 0; i < list.size(); i++){
                Map<String,Pair<Double,Double>> map = list.get(i).get(docId);
                if(map.containsKey(codeSet)){
                    Pair<Double,Double> pair = map.get(codeSet);
                    confidenceSum += pair.getFirst();

                }

            }
            listResult.add(new Pair<>(codeSet,(confidenceSum*1.0)/list.size()));

        }


        List<Pair<String,Double>>finalList = listResult.stream().sorted(comparator.reversed()).collect(Collectors.toList());

        return finalList.get(0);
    }




}
