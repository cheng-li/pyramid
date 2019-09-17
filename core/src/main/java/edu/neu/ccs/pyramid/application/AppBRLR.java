package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.nio.file.Paths;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class AppBRLR {
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
        Config brConfig = createBRLRENConfig(config);
        Config calConfig = createBRCalibrationConfig(config);
        Config predictConfig = createBRPredictionConfig(config);
        Config autoConfig = createBRAutomationConfig(config);
        App1.main(app1Config);
        BRLREN.main(brConfig);
        BRCalibration.main(calConfig);
        BRPrediction.reportValid(predictConfig);
        BRAutomation.tuneThreshold(autoConfig);
        BRPrediction.reportTest(predictConfig);
        BRAutomation.showTestPerformance(autoConfig);


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
        Config brConfig = createBRLRENConfig(config);
        Config calConfig = createBRCalibrationConfig(config);
        Config predictConfig = createBRPredictionConfig(config);
        Config autoConfig = createBRAutomationConfig(config);
        App1.main(app1Config);
        BRLREN.main(brConfig);
        BRCalibration.main(calConfig);
        BRPrediction.reportValid(predictConfig);
        BRAutomation.tuneThreshold(autoConfig);
        BRPrediction.reportTest(predictConfig);
        BRAutomation.showTestPerformance(autoConfig);
    }

    private static Config createApp1Config(Config config){
        Config app1Config = new Config();
        String[] same = {"output.folder","output.trainFolder","output.testFolder","output.calibrationFolder","output.validFolder","output.log",
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
                "train.splitQuery","test.splitQuery","valid.splitQuery", "calibration.splitQuery",
                "train.feature.ngram.matchScoreType","createTrainSet","createTestSet","createValidSet","createCalibrationSet",
                "train.feature.ngram.selection", "train.feature.ngram.selectPerLabel",
                "train.label.order","train.useInstanceWeights","train.weight.field"

        };

        Config.copyExisting(config,app1Config,same);
        return app1Config;
    }


    private static Config createBRLRENConfig(Config config){
        Config brConfig = new Config();

        brConfig.setString("input.trainData", Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.trainFolder")).toString());
        brConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        brConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.validFolder")).toString());
        brConfig.setString("input.calibrationData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.calibrationFolder")).toString());
        brConfig.setString("output.dir",config.getString("output.folder"));
        brConfig.setString("output.verbose","true");
        brConfig.setString("output.log",config.getString("output.log"));

        brConfig.setString("train",config.getString("train"));
        brConfig.setString("predict.allowEmpty","auto");
        brConfig.setString("predict.piThreshold","0.001");


        brConfig.setString("train.maxNumLinearRegUpdates","1");

        brConfig.setString("train.penalty",config.getString("train.penalty"));

        brConfig.setString("train.l1Ratio",config.getString("train.l1Ratio"));
        brConfig.setString("train.numComponents","1");
        brConfig.setString("train.randomInitialize","false");
        brConfig.setString("train.elasticnet.lineSearch","false");
        brConfig.setString("train.elasticnet.activeSet","true");
        //todo
        brConfig.setString("train.updatesPerIteration","1");
        brConfig.setString("train.skipDataThreshold","0.00001");
        brConfig.setString("train.skipLabelThreshold","0.00001");
        brConfig.setString("train.smoothStrength","0.0001");
        Config.copy(config,brConfig,"train.useInstanceWeights");
        Config.copy(config, brConfig,"output.modelFolder");

        return brConfig;
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

        calConfig.setString("validate",config.getString("validate"));
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
        Config.copy(config, calConfig, "threshold.targetValue");
        Config.copy(config, calConfig, "threshold.targetMetric");
        Config.copy(config,calConfig,"threshold.name");
        Config.copy(config, calConfig, "threshold.lowerBound");
        Config.copy(config, calConfig, "threshold.upperBound");
        calConfig.setString("dataSetType","sparse_sequential");

        return calConfig;
    }

    private static Config createBRPredictionConfig(Config config){
        Config predictConfig = new Config();
        predictConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.validFolder")).toString());
        predictConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        predictConfig.setString("input.calibrationData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.calibrationFolder")).toString());
        predictConfig.setString("output.dir",config.getString("output.folder"));
        predictConfig.setString("test",config.getString("test"));
        predictConfig.setString("validate",config.getString("validate"));
        predictConfig.setString("output.log",config.getString("output.log"));

        Config.copy(config,predictConfig,"report.labelSetLimit");
        Config.copy(config,predictConfig,"output.calibratorFolder");
        Config.copy(config,predictConfig,"predict.mode");
        Config.copy(config,predictConfig,"predict.minSize");
        Config.copy(config,predictConfig,"predict.maxSize");
        predictConfig.setString("labelCalibrator",config.getString("calibrate.labelCalibrator"));
        predictConfig.setString("setCalibrator",config.getString("calibrate.setCalibrator"));
        Config.copy(config, predictConfig,"output.modelFolder");
        Config.copy(config, predictConfig, "threshold.targetValue");
        Config.copy(config, predictConfig, "threshold.targetMetric");
        config.copy(config,predictConfig,"threshold.name");
        Config.copy(config, predictConfig, "threshold.lowerBound");
        Config.copy(config, predictConfig, "threshold.upperBound");
        Config.copy(config, predictConfig,"report.showPredictionDetail");
        Config.copy(config, predictConfig,"report.rule.limit");
        Config.copy(config, predictConfig,"report.numDocsPerFile");
        Config.copy(config, predictConfig,"report.classProbThreshold");
        predictConfig.setString("report.produceHTML","false");
        predictConfig.setString("dataSetType","sparse_sequential");

        return predictConfig;
    }


    private static Config createBRAutomationConfig(Config config){
        Config automationConfig = new Config();
        automationConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.validFolder")).toString());
        automationConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        automationConfig.setString("input.calibrationData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.calibrationFolder")).toString());
        automationConfig.setString("output.dir",config.getString("output.folder"));
        automationConfig.setString("test",config.getString("test"));
        automationConfig.setString("validate",config.getString("validate"));
        automationConfig.setString("output.log",config.getString("output.log"));

        Config.copy(config, automationConfig,"output.modelFolder");
        Config.copy(config, automationConfig, "threshold.targetValue");
        Config.copy(config, automationConfig, "threshold.targetMetric");
        config.copy(config,automationConfig,"threshold.name");
        Config.copy(config, automationConfig, "threshold.lowerBound");
        Config.copy(config, automationConfig, "threshold.upperBound");
        Config.copy(config, automationConfig, "tuneThreshold");


        return automationConfig;
    }
}
