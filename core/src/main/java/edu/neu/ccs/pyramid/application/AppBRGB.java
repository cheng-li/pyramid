package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.nio.file.Paths;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;


/**
 * a wrapper that chains App1 and App2
 * Created by chengli on 6/13/15.
 */
public class AppBRGB {
    public static void main(String[] args) throws Exception{
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
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        logger.info(config.toString());
        if (fileHandler!=null){
            fileHandler.close();
        }

        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        Config app1Config = createApp1Config(config);
        Config app2Config = createGBConfig(config);
        Config calConfig = createBRCalibrationConfig(config);
        Config predictConfig = createBRPredictionConfig(config);

        App1.main(app1Config);
        BRGB.main(app2Config);
        BRCalibration.main(calConfig);
        BRPrediction.main(predictConfig);
    }

    public static void main(Config config) throws Exception{
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

        logger.info(config.toString());
        if (fileHandler!=null){
            fileHandler.close();
        }

        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        Config app1Config = createApp1Config(config);
        Config app2Config = createGBConfig(config);
        Config calConfig = createBRCalibrationConfig(config);
        Config predictConfig = createBRPredictionConfig(config);

        App1.main(app1Config);
        BRGB.main(app2Config);
        BRCalibration.main(calConfig);
        BRPrediction.main(predictConfig);
    }

    private static Config createApp1Config(Config config){
        Config app1Config = new Config();
        String[] same = {"output.folder","output.trainFolder","output.testFolder","output.validFolder","output.calibrationFolder","output.log",
                "train.feature.useInitialFeatures","train.feature.categFeature.filter",
                "train.feature.categFeature.percentThreshold","train.feature.ngram.n","train.feature.ngram.minDf","train.feature.ngram.slop",
                "train.feature.missingValue",
                "train.feature.addExternalNgrams","train.feature.externalNgramFile","train.feature.analyzer",
                "train.feature.ngram.allowDuplicateWords","train.feature.ngram.inOrder",
                "train.feature.filterNgramsByKeyWords","train.feature.filterNgrams.keyWordsFile",
                "train.feature.filterNgramsByRegex", "train.feature.filterNgrams.regex",
                "train.feature.useCodeDescription", "train.feature.codeDesc.File", "train.feature.codeDesc.analyzer",
                "train.feature.codeDesc.matchField", "train.feature.codeDesc.minMatchPercentage","test.considerNewLabel","valid.considerNewLabel","calibration.considerNewLabel","train.label.minDF",
                "index.indexName","index.clusterName","index.documentType","index.clientType",
                "index.hosts","index.ports","train.label.field","train.label.filterByPrefix","train.label.filter.prefix",
                "train.feature.featureFieldPrefix","train.feature.ngram.extractionFields",
                "train.splitQuery","test.splitQuery","valid.splitQuery","calibration.splitQuery",
                "train.feature.ngram.matchScoreType","createTrainSet","createTestSet","createValidSet","createCalibrationSet",
                "train.feature.ngram.selection", "train.feature.ngram.selectPerLabel",
                "train.label.order","train.useInstanceWeights","train.weight.field","train.feature.normalize"

        };

        Config.copyExisting(config,app1Config,same);
        return app1Config;
    }


    private static Config createGBConfig(Config config){
        Config gbConfig = new Config();
        String[] same = {"output.folder","output.log", "train","train.warmStart","train.usePrior",
        "train.numIterations","train.numLeaves","train.learningRate","train.minDataPerLeaf",
        "train.numSplitIntervals","train.batchSize", "train.minibatchLifeSpan", "train.fullScanInterval", "train.numActiveFeatures",
                "train.showTrainProgress","train.showValidProgress", "train.showProgress.sampleSize",
                "train.earlyStop.patience","train.earlyStop.minIterations","train.earlyStop",
                "train.earlyStop.absoluteChange", "train.earlyStop.relativeChange",
                "train.showProgress.interval","train.randomSeed","train.useInstanceWeights",
                "output.modelFolder"};

        Config.copyExisting(config,gbConfig,same);

        gbConfig.setString("input.folder",config.getString("output.folder"));
        gbConfig.setString("input.trainData",config.getString("output.trainFolder"));
        gbConfig.setString("input.testData",config.getString("output.testFolder"));
        gbConfig.setString("input.validData",config.getString("output.validFolder"));

        return gbConfig;
    }


    private static Config createBRCalibrationConfig(Config config){
        Config calConfig = new Config();
        calConfig.setString("input.trainData", Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.trainFolder")).toString());
        calConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        calConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets", config.getString("output.validFolder")).toString());
        calConfig.setString("input.calibrationData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.calibrationFolder")).toString());
        calConfig.setString("output.dir",config.getString("output.folder"));
        calConfig.setString("input.calibrationFolder",config.getString("output.calibrationFolder"));
        calConfig.setString("input.validFolder",config.getString("output.validFolder"));
        calConfig.setString("input.testFolder",config.getString("output.testFolder"));
        calConfig.setString("calibrate",config.getString("calibrate"));
        calConfig.setString("test",config.getString("test"));
        calConfig.setString("tuneCTAT",config.getString("tuneCTAT"));
        calConfig.setString("output.log",config.getString("output.log"));
        calConfig.setString("setPrior","true");
        calConfig.setString("brProb","true");
        calConfig.setString("card","true");
        calConfig.setString("encodeLabel","true");
        calConfig.setString("numCandidates",config.getString("calibrate.numCandidates"));
        calConfig.setString("numLeaves",config.getString("calibrate.reranker.numLeaves"));
        calConfig.setString("useInitialFeatures",config.getString("calibrate.reranker.useInitialFeatures"));
        calConfig.setString("featureFieldPrefix",config.getString("calibrate.reranker.featureFieldPrefix"));
        calConfig.setEqual("monotonic=true");
        Config.copy(config,calConfig,"output.calibratorFolder");
        Config.copy(config,calConfig,"predict.mode");
        Config.copy(config,calConfig,"predict.minSize");
        Config.copy(config,calConfig,"predict.maxSize");
        calConfig.setString("labelCalibrator",config.getString("calibrate.labelCalibrator"));
        calConfig.setString("setCalibrator",config.getString("calibrate.setCalibrator"));
        Config.copy(config, calConfig,"output.modelFolder");
        Config.copy(config, calConfig, "CTAT.targetAccuracy");
        Config.copy(config, calConfig, "CTAT.name");
        Config.copy(config, calConfig, "CTAT.lowerBound");
        Config.copy(config, calConfig, "CTAT.upperBound");
        calConfig.setString("dataSetType","sparse_random");

        return calConfig;
    }

    private static Config createBRPredictionConfig(Config config){
        Config predictConfig = new Config();
        predictConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.validFolder")).toString());
        predictConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        predictConfig.setString("output.dir",config.getString("output.folder"));
        predictConfig.setString("test",config.getString("test"));
        predictConfig.setString("validate",config.getString("validate"));
        predictConfig.setString("output.log",config.getString("output.log"));


        Config.copy(config,predictConfig,"output.calibratorFolder");
        Config.copy(config,predictConfig,"predict.mode");
        Config.copy(config,predictConfig,"predict.minSize");
        Config.copy(config,predictConfig,"predict.maxSize");
        predictConfig.setString("labelCalibrator",config.getString("calibrate.labelCalibrator"));
        predictConfig.setString("setCalibrator",config.getString("calibrate.setCalibrator"));
        Config.copy(config, predictConfig,"output.modelFolder");
        Config.copy(config, predictConfig, "CTAT.targetAccuracy");
        Config.copy(config, predictConfig, "CTAT.name");
        Config.copy(config, predictConfig, "CTAT.lowerBound");
        Config.copy(config, predictConfig, "CTAT.upperBound");
        Config.copy(config, predictConfig,"report.showPredictionDetail");
        Config.copy(config, predictConfig,"report.rule.limit");
        Config.copy(config, predictConfig,"report.numDocsPerFile");
        Config.copy(config,predictConfig,"report.labelSetLimit");
        Config.copy(config, predictConfig,"report.classProbThreshold");
        Config.copy(config, predictConfig,"report.produceHTML");

        predictConfig.setString("dataSetType","sparse_random");

        return predictConfig;
    }




}
