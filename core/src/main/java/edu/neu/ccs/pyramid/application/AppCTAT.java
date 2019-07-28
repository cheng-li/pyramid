package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTAT;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.ReportUtils;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Stream;

public class AppCTAT {


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
        logger.info("start tuning CTAT ");

        String validReportPath = config.getString("validReportPath");
        String testReportPath = config.getString("testReportPath");
        Stream<Pair<Double, Double>> validStream = ReportUtils.getConfidenceCorrectness(validReportPath).stream();
        List<Pair<Double, Double>> testList = ReportUtils.getConfidenceCorrectness(testReportPath);

        CTAT.Summary summaryValid = CTAT.findThreshold(validStream,config.getDouble("CTAT.targetAccuracy"));
        double ctat = summaryValid.getConfidenceThreshold();
        double ctat_clipped = ctat;
        if(ctat_clipped > config.getDouble("CTAT.upperBound")){
            ctat_clipped = config.getDouble("CTAT.upperBound");
        }

        if(ctat_clipped < config.getDouble("CTAT.lowerBound")){
            ctat_clipped = config.getDouble("CTAT.lowerBound");
        }

        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),config.getString("CTAT.name")+"_unclipped").toFile(),""+ctat);
        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),config.getString("CTAT.name")+"_clipped").toFile(),""+ctat_clipped);


        CTAT.Summary summaryTest = CTAT.applyThreshold(testList.stream(),ctat);
        CTAT.Summary summaryTest_clipped = CTAT.applyThreshold(testList.stream(),ctat_clipped);

        logger.info("tuning CTAT is done");

        logger.info("*****************");


        logger.info("autocoding performance with unclipped CTAT "+summaryTest.getConfidenceThreshold());
        logger.info("autocoding percentage = "+ summaryTest.getAutoCodingPercentage());
        logger.info("autocoding accuracy = "+ summaryTest.getAutoCodingAccuracy());
        logger.info("number of autocoded documents = "+ summaryTest.getNumAutoCoded());
        logger.info("number of correct autocoded documents = "+ summaryTest.getNumCorrectAutoCoded());

        logger.info("*****************");

        logger.info("autocoding performance with clipped CTAT "+summaryTest_clipped.getConfidenceThreshold());
        logger.info("autocoding percentage = "+ summaryTest_clipped.getAutoCodingPercentage());
        logger.info("autocoding accuracy = "+ summaryTest_clipped.getAutoCodingAccuracy());
        logger.info("number of autocoded documents = "+ summaryTest_clipped.getNumAutoCoded());
        logger.info("number of correct autocoded documents = "+ summaryTest_clipped.getNumCorrectAutoCoded());




        if (fileHandler!=null){
            fileHandler.close();
        }

















    }
























}
