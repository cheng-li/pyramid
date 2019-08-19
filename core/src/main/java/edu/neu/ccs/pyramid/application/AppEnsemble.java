package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTAT;
import edu.neu.ccs.pyramid.calibration.CTFT;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.eval.Recall;
import edu.neu.ccs.pyramid.util.ArgMax;
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
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class AppEnsemble {

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
        List<String> modelNames = config.getStrings("modelNames");
        String ensembleName = config.getString("ensembleModelName");
        String testFolder = config.getString("testFolder");
        String validFolder = config.getString("validFolder");
        double targetValue = config.getDouble("threshold.targetValue");


        logger.info("start loading all reports and getting ground truth");
        List<Map<String,DocumentReport>> testlistMaps = new ArrayList<>();
        List<Map<String,DocumentReport>> validlistMaps = new ArrayList<>();
        Map<String,String> groundTruthTest;
        Map<String,String> groundTruthValid;

        String dataSetPath = modelPaths.get(0).split("model_predictions")[0]+"data_sets/";
        String testSetPath = dataSetPath+testFolder;
        String validSetPath = dataSetPath+validFolder;
        MultiLabelClfDataSet testSetModel0 = TRECFormat.loadMultiLabelClfDataSet(testSetPath,DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet validSetModel0 = TRECFormat.loadMultiLabelClfDataSet(validSetPath,DataSetType.ML_CLF_SPARSE,true);



        groundTruthTest = ReportUtils.getIDGroundTruth(testSetModel0);
        groundTruthValid = ReportUtils.getIDGroundTruth(validSetModel0);

        for(int i = 0; i< modelPaths.size(); i++){

            Map<String,DocumentReport> testmap = loadReportCSV(Paths.get(modelPaths.get(i),"predictions",testFolder+"_reports","report.csv").toString(),modelNames.get(i));
            testlistMaps.add(testmap);

            Map<String,DocumentReport> validmap = loadReportCSV(Paths.get(modelPaths.get(i),"predictions",validFolder+"_reports","report.csv").toString(),modelNames.get(i));
            validlistMaps.add(validmap);

        }
        logger.info("finish loading all reports and getting ground truth");



        logger.info("start generating ensemble test report");
        LabelTranslator newLabelTranslatorTest= getLabelTranslatorEnsemble(config,testFolder);
        List<String> testDocIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",testFolder+"_reports","report.csv").toString());
        generateReport(config,groundTruthTest,testlistMaps,ensembleName,testFolder, testDocIds,newLabelTranslatorTest);
        logger.info("ensemble test report generated");

        logger.info("start generating ensemble validation report");
        LabelTranslator newLabelTranslatorValid= getLabelTranslatorEnsemble(config,validFolder);
        List<String> validDocIds = ReportUtils.getDocIds(Paths.get(modelPaths.get(0),"predictions",validFolder+"_reports","report.csv").toString());
        generateReport(config,groundTruthValid,validlistMaps,ensembleName,validFolder,validDocIds,newLabelTranslatorValid);
        logger.info("ensemble validation report generated");



        logger.info("classification performance on dataset "+testFolder);
        MlMeasureInfo measureInfo_test = getmlMeasureInfo(config,testSetModel0,testFolder,newLabelTranslatorTest);
        MLMeasures mlMeasures = new MLMeasures(measureInfo_test.numClasses,measureInfo_test.multiLabels, measureInfo_test.predictions);
        logger.info(mlMeasures.toString());



        if(config.getBoolean("tuneThreshold")){

            logger.info("start tuning confidence threshold");
            Stream<Pair<Double,Double>> streamValid;
            double threshold = 1.1;
            if(config.getString("threshold.targetMetric").equals("accuracy")){

                streamValid = ReportUtils.getConfidenceCorrectness(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",validFolder+"_reports","report.csv").toString()).stream();
                CTAT.Summary validSummary = CTAT.findThreshold(streamValid,targetValue);
                threshold = validSummary.getConfidenceThreshold();

            }

            if(config.getString("threshold.targetMetric").equals("f1")){
                streamValid =  ReportUtils.getConfidenceF1(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",validFolder+"_reports","report.csv").toString()).stream();
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
                testStream = ReportUtils.getConfidenceCorrectness(Paths.get(config.getString("output.folder"),"model_predictions",ensembleName,"predictions",testFolder+"_reports","report.csv").toString());
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


    public static ESIndex loadIndex(Config config) throws Exception{



        ESIndex.Builder builder = new ESIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))

                .setDocumentType(config.getString("index.documentType"));






        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        ESIndex index = builder.build();
        return index;
    }

    public static Map<String,String> loadSplitVariable(Config config, List<String> docIds) throws Exception{
        String key = config.getString("variableName");
        Map<String,String> map = new HashMap<>();
        try (ESIndex index = loadIndex(config)){
            List<String> variableValues = docIds.stream().parallel().map(docid->{
                List<String> v = index.getStringListField(docid,key);
                if (v.size()>0){
                    return v.get(0);
                }
                else {
                    return "NA";
                }
            }).collect(Collectors.toList());
            for (int i=0;i<docIds.size();i++){
                map.put(docIds.get(i),variableValues.get(i));
            }
        }
        return map;
    }





    private static MlMeasureInfo getmlMeasureInfo(Config config, MultiLabelClfDataSet dataSet,String dataSetFolder,LabelTranslator newLabelTranslator)throws Exception{

        String ensembleReportPath = Paths.get(config.getString("output.folder"),"model_predictions",config.getString("ensembleModelName"),"predictions",dataSetFolder+"_reports","report.csv").toString();
        MultiLabel[] multiLabels = dataSet.getMultiLabels();


        List<String> modelPaths = config.getStrings("modelPaths");

        LabelTranslator labelTranslator0 = (LabelTranslator) Serialization.deserialize(Paths.get(modelPaths.get(0).split("model_predictions")[0],"data_sets",dataSetFolder,"label_translator.ser").toFile());

        MultiLabel[] newMultilabels = new MultiLabel[multiLabels.length];

        MultiLabel[] predictions = ReportUtils.getMultilabelTypePredictions(dataSet,newLabelTranslator,ensembleReportPath);
        for(int i = 0; i < multiLabels.length; i++){
            MultiLabel multiLabel = ReportUtils.reEncodeLabels(multiLabels[i],labelTranslator0,newLabelTranslator);
            newMultilabels[i] = multiLabel;
        }

        MlMeasureInfo measureInfo = new MlMeasureInfo(newMultilabels,predictions,newLabelTranslator.getNumClasses());

        return measureInfo;


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





    private static void generateReport(Config config,Map<String, String> groundTruth, List<Map<String,DocumentReport>> listMaps, String ensembleModelName, String dataSetFolder, List<String> docIds,LabelTranslator labelTranslator)throws Exception{


        StringBuilder sb = new StringBuilder();
        sb.append("doc_id").append("\t").append("prediction").append("\t").append("prediction_type").append("\t")
                .append("confidence").append("\t").append("truth").append("\t").append("ground_truth").append("\t")
                .append("precison").append("\t").append("recall").append("\t").append("F1").append("\t")
                .append("model").append("\n");



        for (int i = 0; i < docIds.size(); i++){
            String docId = docIds.get(i);


            List<DocumentReport> documentReports = new ArrayList<>();


            for (int j = 0; j < listMaps.size(); j++){
                Map<String,DocumentReport> map = listMaps.get(j);
                documentReports.add(map.get(docId));
            }
            List<Double> confidences = documentReports.stream().map(documentReport -> Double.parseDouble(documentReport.setPrediction.confidence)).collect(Collectors.toList());

            int maxIndex = ArgMax.argMax(confidences);
            DocumentReport documentReportMax = documentReports.get(maxIndex);
            ReportLineInfo setPredictionMax = documentReportMax.setPrediction;
            List<ReportLineInfo> labelPredictionsMax = documentReportMax.labelPredictions;

            String[] predictions = setPredictionMax.prediction.split(",");
            String[] labels = groundTruth.get(docId).split(",");
            MultiLabel pre = new MultiLabel();
            MultiLabel lab = new MultiLabel();

            for(String prediction:predictions){
                pre.addLabel(labelTranslator.toIntLabel(prediction));
            }
            for(String label : labels){

                lab.addLabel(labelTranslator.toIntLabel(label.trim()));
            }

            double precision = Precision.precision(lab,pre);
            double recall = Recall.recall(lab,pre);
            double f1 = FMeasure.f1(precision,recall);


            sb.append(setPredictionMax.docId).append("\t").append(sortLabels(setPredictionMax.prediction)).append("\t")
                    .append(setPredictionMax.predictionType).append("\t").append(setPredictionMax.confidence).append("\t")
                    .append(setPredictionMax.correctness).append("\t").append(groundTruth.get(docId)).append("\t")
                    .append(precision).append("\t").append(recall).append("\t").append(f1).append("\t")
                    .append(setPredictionMax.modelName).append("\n");

            for (int k = 0; k < labelPredictionsMax.size(); k++){
                ReportLineInfo reportLineInfoLabel = labelPredictionsMax.get(k);
                sb.append(reportLineInfoLabel.docId).append("\t").append(reportLineInfoLabel.prediction).append("\t")
                        .append(reportLineInfoLabel.predictionType).append("\t").append(reportLineInfoLabel.confidence).append("\t")
                        .append(reportLineInfoLabel.correctness).append("\t").append("NA").append("\t").append("NA").append("\t").append("NA").append("\t").append("NA").append("\t").append(reportLineInfoLabel.modelName).append("\n");

            }

        }


        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),"model_predictions",ensembleModelName,"predictions",dataSetFolder+"_reports","report.csv").toFile(),sb.toString());

    }







    private static String sortLabels(String string){
        String[] split = string.split(",");
        List<String> list = new ArrayList<>();
        for (String str: split){
            String trimmed = str.trim();
            if (!trimmed.isEmpty()){
                list.add(trimmed);
            }
        }
        Collections.sort(list);
        return list.toString().replaceAll("\\]","").replaceAll("\\[","");
    }







    public static class ReportLineInfo {
        public String docId;
        public String prediction;
        public String predictionType;
        public String confidence;
        public String correctness;
        public String modelName;

        public ReportLineInfo(String line, String modelName) {
            String[] split = line.split("\t");
            this.docId = split[0];
            this.prediction = split[1];
            this.predictionType = split[2];
            this.confidence = split[3];
            this.correctness = split[4];
            this.modelName = modelName;
        }
    }

    public static class DocumentReport{
        public ReportLineInfo setPrediction;
        public List<ReportLineInfo> labelPredictions;

        public DocumentReport() {
            this.labelPredictions = new ArrayList<>();
        }

        public void addLine(ReportLineInfo reportLineInfo){
            if (reportLineInfo.predictionType.equals("set")){
                this.setPrediction = reportLineInfo;
            } else {
                this.labelPredictions.add(reportLineInfo);
            }
        }

    }


    public static Map<String,DocumentReport> loadReportCSV(String reportFile, String modelName) throws Exception{

        Map<String,DocumentReport> map = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(reportFile));
        for (String line: lines){
            ReportLineInfo reportLineInfo = new ReportLineInfo(line,modelName);
            if (!map.containsKey(reportLineInfo.docId)){
                map.put(reportLineInfo.docId,new DocumentReport());
            }
            map.get(reportLineInfo.docId).addLine(reportLineInfo);
        }
        return map;
    }


    public static class MlMeasureInfo{
        public MultiLabel[] multiLabels;
        public MultiLabel[] predictions;
        public int numClasses;


        public MlMeasureInfo(MultiLabel[] multiLabels,MultiLabel[] predictions,int numClasses){

            this.multiLabels = multiLabels;
            this.predictions = predictions;
            this.numClasses = numClasses;


        }







    }









}
