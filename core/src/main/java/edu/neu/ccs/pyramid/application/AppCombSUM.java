package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTAT;
import edu.neu.ccs.pyramid.calibration.CTFT;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.eval.Recall;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.FileUtil;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.ReportUtils;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import sun.rmi.runtime.Log;

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



        if (config.getBoolean("calibrate")){
            calibrate(config,logger);
        }


        if (config.getBoolean("validate")){
            report(config,config.getString("validFolder"));
        }


        if(config.getBoolean("tuneThreshold")){
            tuneThreshold(config,logger);

        }

        if (config.getBoolean("test")){
            report(config,config.getString("testFolder"));
            testAutomation(config,logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }

    }


//    private static IsotonicRegression trainIsoRegression(Map<String,Pair<MultiLabel,Double>> map,Config config,LabelTranslator labelTranslator,Map<String,String> groundTruth)throws Exception {
//        List<String> modelPaths = config.getStrings("modelPaths");
//        String calibFolder = config.getString("calibFolder");
//        List<String> docIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",calibFolder+"_reports","report.csv").toString());
//        double[] locations = new double[docIds.size()];
//        double[] numbers = new double[docIds.size()];
//        for (int i = 0; i < docIds.size(); i++) {
//            String docId = docIds.get(i);
//            Pair<MultiLabel, Double> docInfo = map.get(docId);
//
//            MultiLabel pre = docInfo.getFirst();
//            MultiLabel lab = new MultiLabel(groundTruth.get(docId),labelTranslator);
//
//            double truth = 0;
//            if (pre.equals(lab)) {
//                truth = 1;
//            }
//
//            double confidence = map.get(docId).getSecond();
//            locations[i] = confidence;
//            numbers[i] = truth;
//
//        }
//
//        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers);
//
//        return isotonicRegression;
//
//    }

    private static IsotonicRegression trainIsoRegression(Map<String,Pair<MultiLabel,Double>> map,Config config,LabelTranslator labelTranslator,Map<String,String> groundTruth)throws Exception {
        List<String> modelPaths = config.getStrings("modelPaths");
        String calibFolder = config.getString("calibFolder");
        List<String> docIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",calibFolder+"_reports","report.csv").toString());
        double[] locations = new double[docIds.size()];
        double[] numbers = new double[docIds.size()];
        for (int i = 0; i < docIds.size(); i++) {
            String docId = docIds.get(i);
            Pair<MultiLabel, Double> docInfo = map.get(docId);

            MultiLabel pre = docInfo.getFirst();
            MultiLabel lab = new MultiLabel(groundTruth.get(docId),labelTranslator);

            double f1 = FMeasure.f1(pre,lab);
            double confidence = map.get(docId).getSecond();
            locations[i] = confidence;
            numbers[i] = f1;

        }

        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers);

        return isotonicRegression;

    }











    private static void generateReport(Map<String,Pair<MultiLabel,Double>> map,Config config,String dataSetFolder,LabelTranslator labelTranslator,Map<String,String> groundTruth,List<Map<String,Pair<Double,Integer>>> confideceRankLists,IsotonicRegression isotonicRegression)throws Exception{
        List<String> modelNames = config.getStrings("modelNames");
        List<String> modelPaths = config.getStrings("modelPaths");
        List<String> docIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",dataSetFolder+"_reports","report.csv").toString());

        StringBuilder sb = new StringBuilder();
        sb.append("doc_id").append("\t").append("prediction").append("\t").append("prediction_type").append("\t")
                .append("confidence").append("\t").append("truth").append("\t").append("ground_truth").append("\t")
                .append("precision").append("\t").append("recall").append("\t").append("F1").append("\t");

        for(int i = 0; i < modelNames.size(); i++){
            sb.append(modelNames.get(i)).append("_confidence").append("\t").append(modelNames.get(i)).append("_rank").append("\t");
        }
        sb.append("\n");



        for (int i = 0; i < docIds.size(); i++) {
            String docId = docIds.get(i);
            Pair<MultiLabel,Double> docInfo = map.get(docId);

            sb.append(docId).append("\t").append(docInfo.getFirst().toStringWithExtLabels(labelTranslator).replaceAll("\\[","").replaceAll("\\]","")).append("\t").append("set").append("\t")
                    .append(isotonicRegression.predict(map.get(docId).getSecond())).append("\t");


            MultiLabel pre = docInfo.getFirst();
            MultiLabel lab = new MultiLabel(groundTruth.get(docId),labelTranslator);


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


    private static Map<String,Pair<Double,Integer>> getConfidenceRank(Map<String,TopSets> allksetsMap,Map<String,Pair<MultiLabel,Double>> predictionMap,List<String> docIds){
        Map<String,Pair<Double,Integer>> resultMap = new HashMap<>();
        for(int i = 0; i < docIds.size(); i++){
            String docId = docIds.get(i);
            TopSets topKsets = allksetsMap.get(docId);
            MultiLabel prediction = predictionMap.get(docId).getFirst();

            if (topKsets.contains(prediction)){
                resultMap.put(docId,new Pair<>(topKsets.getConfidence(prediction),topKsets.getRank(prediction)));
            } else {
                resultMap.put(docId,new Pair<>(0.0,0));
            }

        }

        return resultMap;



    }



    private static Map<String,TopSets> getTopKset(String reportPath,int limit, LabelTranslator labelTranslator)throws Exception{

        Map<String,TopSets> mapDocsSets = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(reportPath));
        lines.stream().forEach(line->{
            String[] lineInfo = line.split("\t");
            String docId = lineInfo[0];
            String pres = lineInfo[1];
            double confi = Double.parseDouble(lineInfo[2]);
            if(!mapDocsSets.containsKey(docId)){
                TopSets topSets = new TopSets();
                mapDocsSets.put(docId,topSets);

            }
            if(mapDocsSets.get(docId).size() < limit) {
                MultiLabel prediction = new MultiLabel(pres,labelTranslator);
                mapDocsSets.get(docId).add(prediction,confi);
            }
        });

        return mapDocsSets;
    }



    private static Map<String,Pair<MultiLabel,Double>> getFinalPrediction(List<Map<String,TopSets>> list){
        List<String> docIds = list.get(0).keySet().stream().sorted().collect(Collectors.toList());
        Map<String,Pair<MultiLabel,Double>> map = new HashMap<>();

        docIds.stream().forEach(docId->{
            map.put(docId,getFinalPredictionDoc(docId,list));
        });

        return map;
    }


    private static Pair<MultiLabel,Double> getFinalPredictionDoc(String docId,List<Map<String,TopSets>> list){

        Set<MultiLabel> candidateSets = new HashSet<>();

        for(int i = 0; i < list.size(); i++){
            if (list.get(i).get(docId)==null){
                System.out.println("list "+i +" does not contain "+docId);
            }
            candidateSets.addAll(list.get(i).get(docId).allSets());
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<MultiLabel,Double>> listResult = new ArrayList<>();
        for(MultiLabel codeSet : candidateSets){

            double confidenceSum = 0;

            for(int i = 0; i < list.size(); i++){
                TopSets topSets = list.get(i).get(docId);
                if(topSets.contains(codeSet)){
                    confidenceSum += topSets.getConfidence(codeSet);
                }

            }
            listResult.add(new Pair<>(codeSet,(confidenceSum*1.0)/list.size()));
        }


        List<Pair<MultiLabel,Double>>finalList = listResult.stream().sorted(comparator.reversed()).collect(Collectors.toList());

        return finalList.get(0);
    }

    private static Map<String,Pair<MultiLabel,Double>> getEnsemblePrediction(Config config, String folderName) throws Exception{
        List<String> modelPaths = config.getStrings("modelPaths");
        LabelTranslator labelTranslator = getLabelTranslatorEnsemble(config,folderName);

        List<Map<String,TopSets>> mapLists = new ArrayList<>();

        int limit = config.getInt("topKSets");
        for(int i = 0; i < modelPaths.size(); i++){
            String reportPath = Paths.get(modelPaths.get(i),"predictions",folderName+"_reports","top_sets.csv").toString();
            mapLists.add(getTopKset(reportPath,limit,labelTranslator));
        }
        return getFinalPrediction(mapLists);
    }

    private static List<Map<String,Pair<Double,Integer>>> getConfidenceRankLists(Config config, String folderName, Map<String,Pair<MultiLabel,Double>> ensemblePrediction) throws Exception{
        List<String> modelPaths = config.getStrings("modelPaths");
        LabelTranslator labelTranslator = getLabelTranslatorEnsemble(config,folderName);
        List<String> docIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",folderName+"_reports","report.csv").toString());
        List<Map<String,Pair<Double,Integer>>> confideceRankLists = new ArrayList<>();
        for(int i = 0; i < modelPaths.size(); i++) {
            String reportPath = Paths.get(modelPaths.get(i),"predictions",folderName+"_reports","top_sets.csv").toString();

            Map<String,TopSets> allKSetsMap = getTopKset(reportPath,20,labelTranslator);
            confideceRankLists.add(getConfidenceRank(allKSetsMap,ensemblePrediction,docIds));
        }
        return confideceRankLists;

    }

    private static Map<String,String> getGroundTruth(Config config, String folderName) throws Exception{
        List<String> modelPaths = config.getStrings("modelPaths");
        String path = modelPaths.get(0).split("model_predictions")[0]+"data_sets/"+folderName;

        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(path, DataSetType.ML_CLF_SPARSE,true);

        Map<String,String> groundTruth= ReportUtils.getIDGroundTruth(dataSet);
        return groundTruth;
    }


    private static void tuneThreshold(Config config, Logger logger) throws Exception{
        logger.info("start tuning confidence threshold");
        Stream<Pair<Double,Double>> streamValid;
        double threshold = 1.1;
        double targetValue = config.getDouble("threshold.targetValue");
        String ensembleName = config.getString("ensembleModelName");
        String validFolder = config.getString("validFolder");
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
    }


    public static LabelTranslator getLabelTranslatorEnsemble(Config config,String dataSetFolder)throws Exception{

        Set<String> allExtLabelsSet = new HashSet<>();
        List<String> modelPaths = config.getStrings("modelPaths");
        for(int i = 0; i < modelPaths.size(); i++){
            LabelTranslator labelTranslator = (LabelTranslator) Serialization.deserialize(Paths.get(modelPaths.get(i).split("model_predictions")[0],"data_sets",dataSetFolder,"label_translator.ser").toFile());
            allExtLabelsSet.addAll(labelTranslator.getAllExtLabels());
        }
        List<String> allExtLabels = allExtLabelsSet.stream().sorted()
                .collect(Collectors.toList());

        LabelTranslator newLabelTranslator = new LabelTranslator(allExtLabels);
        return newLabelTranslator;



    }
    
    private static void calibrate(Config config, Logger logger) throws Exception{
        String calibFolder = config.getString("calibFolder");
        LabelTranslator newLabelTranslatorCalib = getLabelTranslatorEnsemble(config,calibFolder);
        Map<String,String> groundTruthCalib = getGroundTruth(config,calibFolder);
        Map<String,Pair<MultiLabel,Double>> calibMap = getEnsemblePrediction(config,calibFolder);
        IsotonicRegression isotonicRegression = trainIsoRegression(calibMap,config,newLabelTranslatorCalib,groundTruthCalib);
        Serialization.serialize(isotonicRegression, Paths.get(config.getString("output.folder"), "model_predictions", config.getString("ensembleModelName"), "models","set_calibrator"));
    }


    private static void report(Config config, String folderName) throws Exception{
        Map<String,Pair<MultiLabel,Double>> map = getEnsemblePrediction(config,folderName);
        LabelTranslator labelTranslatorEnsemble = getLabelTranslatorEnsemble(config,folderName);
        Map<String,String> groundTruth = getGroundTruth(config,folderName);
        List<Map<String,Pair<Double,Integer>>> confidenceRankLists = getConfidenceRankLists(config,folderName,map);
        IsotonicRegression isotonicRegression = (IsotonicRegression)Serialization.deserialize(Paths.get(config.getString("output.folder"), "model_predictions", config.getString("ensembleModelName"), "models","set_calibrator"));
        generateReport(map,config,folderName,labelTranslatorEnsemble,groundTruth,confidenceRankLists,isotonicRegression);
    }


    private static void testAutomation(Config config, Logger logger) throws Exception{
        List<Pair<Double,Double>> testStream;
        String ensembleName = config.getString("ensembleModelName");
        String testFolder = config.getString("testFolder");
        double threshold = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"models",
                "threshold",config.getString("threshold.name")).toFile()));
        double confidenceThresholdClipped = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"models",
                "threshold",config.getString("threshold.name")+"_clipped").toFile()));

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

    
    private static class TopSets{
        private Map<MultiLabel,Double> predictionToConfidence;
        private Map<MultiLabel,Integer> predictionToRank;

        public TopSets() {
            predictionToConfidence = new HashMap<>();
            predictionToRank = new HashMap<>();
        }

        void add(MultiLabel multiLabel, double confidence){
            int rank = predictionToConfidence.size()+1;
            predictionToConfidence.put(multiLabel,confidence);
            predictionToRank.put(multiLabel,rank);
        }
        
        int size(){
            return predictionToConfidence.size();
        }

        Set<MultiLabel> allSets(){
            return predictionToConfidence.keySet();
        }

        boolean contains(MultiLabel multiLabel){
            return predictionToConfidence.containsKey(multiLabel);
        }

        double getConfidence(MultiLabel multiLabel){
            return predictionToConfidence.get(multiLabel);
        }

        int getRank(MultiLabel multiLabel){
            return predictionToRank.get(multiLabel);
        }
        
    }


}
