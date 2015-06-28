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
            throw new IllegalArgumentException("please specify the config file");
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
        String[] same = {"output.folder","feature.useInitialFeatures","feature.categFeature.filter",
                "feature.categFeature.percentThreshold","feature.ngram.n","feature.ngram.minDf","feature.ngram.slop",
                "feature.missingValue","index.indexName","index.clusterName","index.documentType","index.clientType",
                "index.hosts","index.ports","index.labelField","index.featureFieldPrefix","index.ngramExtractionFields",
                "index.splitField","index.splitField.train","index.splitField.test"};

        Config.copy(config,app1Config,same);

        app1Config.setBoolean("createTrainSet",config.getBoolean("train"));
        if (config.getBoolean("train.warmStart")){
            app1Config.setBoolean("createTrainSet",false);
        }
        app1Config.setBoolean("createTestSet",config.getBoolean("test"));
        return app1Config;
    }


    private static Config createApp2Config(Config config){
        Config app2Config = new Config();
        String[] same = {"output.folder","train","test","train.prediction.fashion","train.warmStart","train.usePrior",
        "train.numIterations","train.numLeaves","train.learningRate","train.minDataPerLeaf","train.featureSamplingRate",
        "train.dataSamplingRate","train.numSplitIntervals","train.showPerformanceEachRound",
        "report.topFeatures.limit","report.rule.limit","report.numDocsPerFile","report.classProbThreshold"};

        Config.copy(config,app2Config,same);

        app2Config.setString("input.folder",config.getString("output.folder"));
        app2Config.setString("input.trainData",config.getString("index.splitField.train"));
        app2Config.setString("input.testData",config.getString("index.splitField.test"));
        return app2Config;
    }


}
