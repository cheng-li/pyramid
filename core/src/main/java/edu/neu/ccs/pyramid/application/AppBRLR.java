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
        Config app2Config = createApp2Config(config);

        App1.main(app1Config);
        App2.main(app2Config);
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
        Config app2Config = createApp2Config(config);

        App1.main(app1Config);
        App2.main(app2Config);
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
                "train.label.order"

        };

        Config.copyExisting(config,app1Config,same);
        return app1Config;
    }


    private static Config createBRLRENConfig(Config config){
        Config brConfig = new Config();
        String[] same = {"test"};

        Config.copyExisting(config,brConfig,same);


        brConfig.setString("input.trainData", Paths.get(config.getString("output.folder"),config.getString("output.trainFolder")).toString());
        brConfig.setString("input.testData",Paths.get(config.getString("output.folder"),config.getString("output.testFolder")).toString());
        brConfig.setString("input.validData",Paths.get(config.getString("output.folder"),config.getString("output.validFolder")).toString());
        brConfig.setString("output.dir",config.getString("output.folder"));
        brConfig.setString("output.verbose","true");
        brConfig.setString("tune",config.getString("train"));
        brConfig.setString("train",config.getString("train"));
        brConfig.setString("predict.allowEmpty","auto");
        brConfig.setString("predict.piThreshold","0.001");
        brConfig.setString("tune.targetMetric","instance_set_accuracy");
        brConfig.setString("tune.penalty.candidates",config.getString("train.penalty.candidates"));
        brConfig.setString("tune.l1Ratio.candidates",config.getString("train.l1Ratio.candidates"));
        brConfig.setString("tune.numComponents.candidates","1");
        brConfig.setString("tune.monitorInterval","1");
        brConfig.setString("tune.earlyStop.minIterations","5");
        brConfig.setString("tune.earlyStop.patience","10");
        



        return brConfig;
    }
}
