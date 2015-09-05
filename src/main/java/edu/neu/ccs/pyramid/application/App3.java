package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.nio.file.Paths;

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
        String[] same = {"output.folder","feature.useInitialFeatures","feature.categFeature.filter",
                "feature.categFeature.percentThreshold","feature.ngram.n","feature.ngram.minDf","feature.ngram.slop",
                "feature.missingValue","index.indexName","index.clusterName","index.documentType","index.clientType",
                "index.hosts","index.ports","index.labelField","index.featureFieldPrefix","index.ngramExtractionFields",
                "index.splitField","index.splitField.train","index.splitField.test",
                "index.ngramMatchScoreType"};

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
        String[] same = {"output.folder","train","test","predict.fashion","train.warmStart","train.usePrior",
        "train.numIterations","train.numLeaves","train.learningRate","train.minDataPerLeaf","train.featureSamplingRate",
        "train.dataSamplingRate","train.numSplitIntervals","train.showTrainProgress","train.showTestProgress",
                "train.showProgress.interval",
        "report.topFeatures.limit","report.rule.limit","report.numDocsPerFile","report.classProbThreshold","report.labelSetLimit"};

        Config.copy(config,app2Config,same);

        app2Config.setString("input.folder",config.getString("output.folder"));
        app2Config.setString("input.trainData",App1.splitListToString(config.getStrings("index.splitField.train")));
        app2Config.setString("input.testData",App1.splitListToString(config.getStrings("index.splitField.test")));
        return app2Config;
    }




}
