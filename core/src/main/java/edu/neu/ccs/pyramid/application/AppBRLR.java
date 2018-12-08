package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import org.apache.hadoop.hdfs.DFSClient;

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
        App1.main(app1Config);
        BRLREN.main(brConfig);
        BRCalibration.main(calConfig);

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
        App1.main(app1Config);
        BRLREN.main(brConfig);
        BRCalibration.main(calConfig);
    }

    private static Config createApp1Config(Config config){
        Config app1Config = new Config();
        String[] same = {"output.folder","output.trainFolder","output.testFolder","output.validFolder","output.log",
                "train.feature.useInitialFeatures","train.feature.categFeature.filter",
                "train.feature.categFeature.percentThreshold","train.feature.ngram.n","train.feature.ngram.minDf","train.feature.ngram.slop",
                "train.feature.missingValue",
                "train.feature.addExternalNgrams","train.feature.externalNgramFile","train.feature.analyzer",
                "train.feature.ngram.allowDuplicateWords","train.feature.ngram.inOrder",
                "train.feature.filterNgramsByKeyWords","train.feature.filterNgrams.keyWordsFile",
                "train.feature.filterNgramsByRegex", "train.feature.filterNgrams.regex",
                "train.feature.useCodeDescription", "train.feature.codeDesc.File", "train.feature.codeDesc.analyzer",
                "train.feature.codeDesc.matchField", "train.feature.codeDesc.minMatchPercentage","test.considerNewLabel","valid.considerNewLabel","train.label.minDF",
                "index.indexName","index.clusterName","index.documentType","index.clientType",
                "index.hosts","index.ports","train.label.field","train.label.filterByPrefix","train.label.filter.prefix",
                "train.feature.featureFieldPrefix","train.feature.ngram.extractionFields",
                "train.splitQuery","test.splitQuery","valid.splitQuery",
                "train.feature.ngram.matchScoreType","createTrainSet","createTestSet","createValidSet",
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
        brConfig.setString("output.dir",config.getString("output.folder"));
        brConfig.setString("output.verbose","true");
        brConfig.setString("output.log",config.getString("output.log"));
        //todo
        brConfig.setString("tune","false");
        brConfig.setString("train",config.getString("train"));
        brConfig.setString("predict.allowEmpty","auto");
        brConfig.setString("predict.piThreshold","0.001");
        brConfig.setString("tune.targetMetric","instance_set_accuracy");
        brConfig.setString("tune.penalty.candidates",config.getString("train.penalty.candidates"));
        brConfig.setString("tune.l1Ratio.candidates",config.getString("train.l1Ratio.candidates"));
        brConfig.setString("tune.numComponents.candidates","1");
        brConfig.setString("tune.monitorInterval","1");
        brConfig.setString("tune.earlyStop.minIterations","3");
        brConfig.setString("tune.earlyStop.patience","3");
        //todo
        brConfig.setString("train.useTunedHyperParameters","false");
        //todo
        brConfig.setString("train.iterations",config.getString("train.iterations"));
        //todo
        brConfig.setString("train.penalty",config.getString("train.penalty"));
        //todo
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
        brConfig.setString("report.labelProbThreshold",config.getString("report.classProbThreshold"));
        brConfig.setString("test","false");
        Config.copy(config,brConfig,"train.useInstanceWeights");
        Config.copy(config, brConfig,"output.modelName");
        return brConfig;
    }


    private static Config createBRCalibrationConfig(Config config){
        Config calConfig = new Config();
        calConfig.setString("input.trainData", Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.trainFolder")).toString());
        calConfig.setString("input.testData",Paths.get(config.getString("output.folder"),"data_sets",config.getString("output.testFolder")).toString());
        calConfig.setString("input.validData",Paths.get(config.getString("output.folder"),"data_sets", config.getString("output.validFolder")).toString());
        calConfig.setString("output.dir",config.getString("output.folder"));
        calConfig.setString("calibrate",config.getString("calibrate"));
        calConfig.setString("test",config.getString("test"));
        calConfig.setString("output.log",config.getString("output.log"));
        calConfig.setString("setPrior","true");
        calConfig.setString("brProb","true");
        calConfig.setString("cardPrior","false");
        calConfig.setString("card","true");
        calConfig.setString("pairPrior","false");
        calConfig.setString("encodeLabel","false");
        calConfig.setString("f1Prior","false");
        calConfig.setString("cbmProb","false");
        calConfig.setString("implication","false");
        calConfig.setEqual("labelProbs=false");
        calConfig.setEqual("position=false");
        calConfig.setString("numCandidates",config.getString("calibrate.numCandidates"));
        calConfig.setString("numIterations",config.getString("calibrate.reranker.numIterations"));
        calConfig.setString("numLeaves",config.getString("calibrate.reranker.numLeaves"));
        calConfig.setEqual("monotonic=true");
        calConfig.setEqual("logScale=false");
        Config.copy(config,calConfig,"report.labelSetLimit");
        Config.copy(config,calConfig,"calibrate.folder");
        Config.copy(config,calConfig,"predict.mode");
        calConfig.setString("labelCalibrator",config.getString("calibrate.labelCalibrator"));
        calConfig.setString("setCalibrator",config.getString("calibrate.setCalibrator"));
        return calConfig;
    }
}
