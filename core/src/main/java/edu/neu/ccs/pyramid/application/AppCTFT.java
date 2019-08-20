package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.CTFT;
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

public class AppCTFT {

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
        logger.info("start tuning CTFT ");

        String validReportPath = config.getString("validReportPath");
        String testReportPath = config.getString("testReportPath");
        Stream<Pair<Double, Double>> validStream = ReportUtils.getConfidenceF1(validReportPath).stream();
        List<Pair<Double, Double>> testList = ReportUtils.getConfidenceF1(testReportPath);

        CTFT.Summary summaryValid = CTFT.findThreshold(validStream,config.getDouble("CTFT.targetF1"));
        double ctft = summaryValid.getConfidenceThreshold();
        double ctft_clipped = ctft;
        if(ctft_clipped > config.getDouble("CTFT.upperBound")){
            ctft_clipped = config.getDouble("CTFT.upperBound");
        }

        if(ctft_clipped < config.getDouble("CTFT.lowerBound")){
            ctft_clipped = config.getDouble("CTFT.lowerBound");
        }

        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),config.getString("CTFT.name")+"_unclipped").toFile(),""+ctft);
        FileUtils.writeStringToFile(Paths.get(config.getString("output.folder"),config.getString("CTFT.name")+"_clipped").toFile(),""+ctft_clipped);


        CTFT.Summary summaryTest = CTFT.applyThreshold(testList.stream(),ctft);
        CTFT.Summary summaryTest_clipped = CTFT.applyThreshold(testList.stream(),ctft_clipped);

        logger.info("tuning CTFT is done");

        logger.info("*****************");


        logger.info("autocoding performance with unclipped CTFT "+summaryTest.getConfidenceThreshold());
        logger.info("autocoding percentage = "+ summaryTest.getAutoCodingPercentage());
        logger.info("autocoding accuracy = "+ summaryTest.getAutoCodingAccuracy());
        logger.info("autocoding F1 = "+ summaryTest.getAutoCodingF1());
        logger.info("number of autocoded documents = "+ summaryTest.getNumAutoCoded());
        logger.info("number of correct autocoded documents = "+ summaryTest.getNumCorrectAutoCoded());

        logger.info("*****************");

        logger.info("autocoding performance with clipped CTFT "+summaryTest_clipped.getConfidenceThreshold());
        logger.info("autocoding percentage = "+ summaryTest_clipped.getAutoCodingPercentage());
        logger.info("autocoding accuracy = "+ summaryTest_clipped.getAutoCodingAccuracy());
        logger.info("autocoding F1 = "+ summaryTest_clipped.getAutoCodingF1());
        logger.info("number of autocoded documents = "+ summaryTest_clipped.getNumAutoCoded());
        logger.info("number of correct autocoded documents = "+ summaryTest_clipped.getNumCorrectAutoCoded());




        if (fileHandler!=null){
            fileHandler.close();
        }

















    }







}
