package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTAT;
import edu.neu.ccs.pyramid.calibration.CTFT;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.ReportUtils;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Paths;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Stream;

public class BRAutomation {


    public static void tuneThreshold(Config config) throws Exception{
        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        if(config.getBoolean("tuneThreshold")){
            tuneThreshold(config,logger);
            showAutomationPerformance(config,config.getString("input.validData"),logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }


    public static void showTestPerformance(Config config) throws Exception{
        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        if(config.getBoolean("test")){
            showAutomationPerformance(config,config.getString("input.testData"),logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    private static void tuneThreshold(Config config, Logger logger)throws Exception{
        String dataPath = config.getString("input.validData");
        File validDataFile = new File(dataPath);
        String reportFolder = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",validDataFile.getName()+"_reports").toString();
        String targetMetric = config.getString("threshold.targetMetric");
        String reportPath = Paths.get(reportFolder,"report.csv").toString();
        Stream<Pair<Double, Double>> validStream;
        double threshold =1.1;
        if(targetMetric.equals("accuracy")) {
            validStream = ReportUtils.getConfidenceCorrectness(reportPath).stream();
            CTAT.Summary summaryValid = CTAT.findThreshold(validStream,config.getDouble("threshold.targetValue"));
            threshold = summaryValid.getConfidenceThreshold();
        }
        if(targetMetric.equals("f1")){
            validStream = ReportUtils.getConfidenceF1(reportPath).stream();
            CTFT.Summary summaryValid = CTFT.findThreshold(validStream,config.getDouble("threshold.targetValue"));
            threshold = summaryValid.getConfidenceThreshold();
        }

        logger.info("confidence threshold for target " +config.getString("threshold.targetMetric")+" "+config.getDouble("threshold.targetValue") +" = " +threshold);

        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "threshold",config.getString("threshold.name")).toFile(),""+threshold);

        double confidenceThresholdClipped = CTAT.clip(threshold,config.getDouble("threshold.lowerBound"),config.getDouble("threshold.upperBound"));

        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "threshold",config.getString("threshold.name")+"_clipped").toFile(),""+confidenceThresholdClipped);

    }

    public static void showAutomationPerformance(Config config, String dataPath, Logger logger) throws Exception{
        double confidenceThreshold = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.dir"), "model_predictions", config.getString("output.modelFolder"), "models",
                "threshold", config.getString("threshold.name")).toFile()));
        File testDataFile = new File(dataPath);
        String reportFolder = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",testDataFile.getName()+"_reports").toString();
        String reportPath = Paths.get(reportFolder,"report.csv").toString();
        if (config.getString("threshold.targetMetric").equals("accuracy")) {
            CTAT.Summary summary = CTAT.applyThreshold(ReportUtils.getConfidenceCorrectness(reportPath).stream(), confidenceThreshold);
            logger.info("autocoding performance with unclipped CTAT " + summary.getConfidenceThreshold());
            logger.info("autocoding percentage = " + summary.getAutoCodingPercentage());
            logger.info("autocoding accuracy = " + summary.getAutoCodingAccuracy());
            logger.info("number of autocoded documents = " + summary.getNumAutoCoded());
            logger.info("number of correct autocoded documents = " + summary.getNumCorrectAutoCoded());

            double confidenceThresholdClipped = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.dir"), "model_predictions", config.getString("output.modelFolder"), "models",
                    "threshold", config.getString("threshold.name") + "_clipped").toFile()));
            CTAT.Summary summaryClipped = CTAT.applyThreshold(ReportUtils.getConfidenceCorrectness(reportPath).stream(), confidenceThresholdClipped);
            logger.info("autocoding performance with clipped CTAT " + summaryClipped.getConfidenceThreshold());
            logger.info("autocoding percentage = " + summaryClipped.getAutoCodingPercentage());
            logger.info("autocoding accuracy = " + summaryClipped.getAutoCodingAccuracy());
            logger.info("number of autocoded documents = " + summaryClipped.getNumAutoCoded());
            logger.info("number of correct autocoded documents = " + summaryClipped.getNumCorrectAutoCoded());
        }

        if (config.getString("threshold.targetMetric").equals("f1")) {

            CTFT.Summary summary = CTFT.applyThreshold(ReportUtils.getConfidenceF1(reportPath).stream(), confidenceThreshold);
            logger.info("autocoding performance with unclipped CTFT " + summary.getConfidenceThreshold());
            logger.info("autocoding percentage = " + summary.getAutoCodingPercentage());
            logger.info("autocoding accuracy = " + summary.getAutoCodingAccuracy());
            logger.info("autocoding F1 = " + summary.getAutoCodingF1());
            logger.info("number of autocoded documents = " + summary.getNumAutoCoded());
            logger.info("number of correct autocoded documents = " + summary.getNumCorrectAutoCoded());

            double confidenceThresholdClipped = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.dir"), "model_predictions", config.getString("output.modelFolder"), "models",
                    "threshold", config.getString("threshold.name") + "_clipped").toFile()));
            CTFT.Summary summaryClipped = CTFT.applyThreshold(ReportUtils.getConfidenceF1(reportPath).stream(), confidenceThresholdClipped);
            logger.info("autocoding performance with clipped CTFT " + summaryClipped.getConfidenceThreshold());
            logger.info("autocoding percentage = " + summaryClipped.getAutoCodingPercentage());
            logger.info("autocoding accuracy = " + summaryClipped.getAutoCodingAccuracy());
            logger.info("autocoding F1 = " + summaryClipped.getAutoCodingF1());
            logger.info("number of autocoded documents = " + summaryClipped.getNumAutoCoded());
            logger.info("number of correct autocoded documents = " + summaryClipped.getNumCorrectAutoCoded());
        }
    }





}
