package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;


/**
 * a wrapper that chains App1 and App2
 * Created by chengli on 6/13/15.
 */
public class App3 {
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


    private static Config createApp2Config(Config config){
        Config app2Config = new Config();
        String[] same = {"output.folder","output.log", "train","calibrate","test","tune","predict.target","train.warmStart","train.usePrior",
        "train.numIterations","train.numLeaves","train.learningRate","train.minDataPerLeaf",
        "train.numSplitIntervals","train.batchSize", "train.minibatchLifeSpan", "train.fullScanInterval", "train.numActiveFeatures",
                "train.showTrainProgress","train.showValidProgress", "train.showProgress.sampleSize",
                "train.earlyStop.patience","train.earlyStop.minIterations","train.earlyStop",
                "train.earlyStop.absoluteChange", "train.earlyStop.relativeChange",
                "train.showProgress.interval","train.generateReports","train.randomSeed","tune.FMeasure.beta", "report.order",
        "report.topFeatures.limit","report.rule.limit","report.numDocsPerFile","report.classProbThreshold","report.labelSetLimit",
                "report.showPredictionDetail","report.produceHTML"};

        Config.copyExisting(config,app2Config,same);

        app2Config.setString("input.folder",config.getString("output.folder"));
        app2Config.setString("input.trainData",config.getString("output.trainFolder"));
        app2Config.setString("input.testData",config.getString("output.testFolder"));
        app2Config.setString("input.validData",config.getString("output.validFolder"));
        return app2Config;
    }




}
