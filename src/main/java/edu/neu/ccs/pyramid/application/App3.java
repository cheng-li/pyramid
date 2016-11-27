package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;


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
        System.out.println(config);
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        Config app1Config = createApp1Config(config);
        Config app2Config = createApp2Config(config);

        App1.main(app1Config);
        App2.main(app2Config);
    }

    private static Config createApp1Config(Config config){
        Config app1Config = new Config();
        String[] same = {"output.folder","output.trainFolder","output.testFolder","output.log",
                "feature.useInitialFeatures","feature.categFeature.filter",
                "feature.categFeature.percentThreshold","feature.ngram.n","feature.ngram.minDf","feature.ngram.slop",
                "feature.missingValue","feature.generateDistribution",
                "feature.addExternalNgrams","feature.externalNgramFile","feature.analyzer",
                "feature.filterNgramsByKeyWords","feature.filterNgrams.keyWordsFile",
                "feature.filterNgramsByRegex", "feature.filterNgrams.regex",
                "feature.useCodeDescription", "feature.codeDesc.File", "feature.codeDesc.analyzer",
                "feature.codeDesc.matchField", "feature.codeDesc.minMatchPercentage",
                "index.indexName","index.clusterName","index.documentType","index.clientType",
                "index.hosts","index.ports","index.labelField","index.labelFilter","index.labelFilter.prefix",
                "index.featureFieldPrefix","index.ngramExtractionFields",
                "index.splitQuery.train","index.splitQuery.test",
                "index.ngramMatchScoreType","createTrainSet","createTestSet"};

        Config.copy(config,app1Config,same);
        return app1Config;
    }


    private static Config createApp2Config(Config config){
        Config app2Config = new Config();
        String[] same = {"output.folder","output.log", "train","test","tune","predict.target","train.warmStart","train.usePrior",
        "train.numIterations","train.numLeaves","train.learningRate","train.minDataPerLeaf",
        "train.numSplitIntervals","train.showTrainProgress","train.showTestProgress",
                "train.earlyStop.patience","train.earlyStop.minIterations","train.earlyStop",
                "train.earlyStop.absoluteChange", "train.earlyStop.relativeChange",
                "train.showProgress.interval","train.generateReports","tune.data","tune.FMeasure.beta",
        "report.topFeatures.limit","report.rule.limit","report.numDocsPerFile","report.classProbThreshold","report.labelSetLimit",
                "report.showPredictionDetail"};

        Config.copy(config,app2Config,same);

        app2Config.setString("input.folder",config.getString("output.folder"));
        app2Config.setString("input.trainData",config.getString("output.trainFolder"));
        app2Config.setString("input.testData",config.getString("output.testFolder"));
        return app2Config;
    }




}
