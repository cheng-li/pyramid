package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.feature_selection.FeatureDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.MacroFMeasureTuner;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.TunedMarginalClassifier;
import edu.neu.ccs.pyramid.optimization.EarlyStopper;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.Progress;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.neu.ccs.pyramid.util.SetUtil;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * imlgb
 * Created by chengli on 6/13/15.
 */
public class App2 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        main(config);
    }

    public static void main(Config config) throws Exception{
        System.out.println(config);

        new File(config.getString("output.folder")).mkdirs();

        if (config.getBoolean("train")){
            train(config);
            if (config.getString("predict.target").equals("macroFMeasure")){
                System.out.println("predict.target=macroFMeasure,  user needs to run 'tune' before predictions can be made. " +
                        "Reports will be generated after tuning.");
            } else {
                if (config.getBoolean("train.generateReports")){
                    report(config,config.getString("input.trainData"));
                }

            }

        }

        if (config.getBoolean("tune")){
            tuneForMacroF(config);
            if (config.getBoolean("train.generateReports")){
                report(config,config.getString("input.trainData"));
            }
        }

        if (config.getBoolean("test")){
            report(config,config.getString("input.testData"));
        }
    }

    static MultiLabelClfDataSet loadData(Config config, String dataName) throws Exception{
        File dataFile = new File(new File(config.getString("input.folder"),
                "data_sets"),dataName);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(dataFile, DataSetType.ML_CLF_SPARSE,
                true);
        return dataSet;
    }

    static void train(Config config) throws Exception{
        String output = config.getString("output.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = "model_app3";
//        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
//        double dataSamplingRate = config.getDouble("train.dataSamplingRate");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadData(config,config.getString("input.trainData"));

        MultiLabelClfDataSet testSet = null;
        if (config.getBoolean("train.showTestProgress")){
            testSet = loadData(config,config.getString("input.testData"));
        }



        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);
        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
//                .dataSamplingRate(dataSamplingRate)
//                .featureSamplingRate(featureSamplingRate)
                .learningRate(learningRate)
                .minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves)
                .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                .usePrior(config.getBoolean("train.usePrior"))
                .build();

        IMLGradientBoosting boosting;
        if (config.getBoolean("train.warmStart")){
            boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        } else {
            boosting  = new IMLGradientBoosting(numClasses);
        }

        System.out.println("During training, the performance is reported using Hamming loss optimal predictor");
        System.out.println("initialing trainer");

        IMLGBTrainer trainer = new IMLGBTrainer(imlgbConfig,boosting);


        boolean earlyStop = config.getBoolean("train.earlyStop");

        List<EarlyStopper> earlyStoppers = new ArrayList<>();
        List<Terminator> terminators = new ArrayList<>();

        if (earlyStop){
            for (int l=0;l<numClasses;l++){
                EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE, config.getInt("train.earlyStop.patience"));
                earlyStopper.setMinimumIterations(config.getInt("train.earlyStop.minIterations"));
                earlyStoppers.add(earlyStopper);
            }


            for (int l=0;l<numClasses;l++){
                Terminator terminator = new Terminator();
                terminator.setMaxStableIterations(config.getInt("train.earlyStop.patience"))
                        .setMinIterations(config.getInt("train.earlyStop.minIterations")/config.getInt("train.showProgress.interval"))
                        .setAbsoluteEpsilon(config.getDouble("train.earlyStop.absoluteChange"))
                        .setRelativeEpsilon(config.getDouble("train.earlyStop.relativeChange"))
                        .setOperation(Terminator.Operation.OR);
                terminators.add(terminator);
            }
        }






        System.out.println("trainer initialized");

        int numLabelsLeftToTrain = numClasses;

        int progressInterval = config.getInt("train.showProgress.interval");
        for (int i=1;i<=numIterations;i++){
            System.out.println("iteration "+i);
            trainer.iterate();
            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("training set performance");
                System.out.println(new MLMeasures(boosting,dataSet));
            }
            if (config.getBoolean("train.showTestProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("test set performance");
                System.out.println(new MLMeasures(boosting,testSet));
                if (earlyStop){
                    for (int l=0;l<numClasses;l++){
                        EarlyStopper earlyStopper = earlyStoppers.get(l);
                        Terminator terminator = terminators.get(l);
                        if (!trainer.getShouldStop()[l]){
                            double kl = KL(boosting, testSet, l);
                            earlyStopper.add(i,kl);
                            terminator.add(kl);
                            if (earlyStopper.shouldStop() || terminator.shouldTerminate()){
                                System.out.println("training for label "+l+" ("+dataSet.getLabelTranslator().toExtLabel(l)+") should stop now");
                                System.out.println("the best number of training iterations for the label is "+earlyStopper.getBestIteration());
                                trainer.setShouldStop(l);
                                numLabelsLeftToTrain -= 1;
                                System.out.println("the number of labels left to be trained on = "+numLabelsLeftToTrain);
                            }
                        }
                    }
                }

            }
            if (numLabelsLeftToTrain==0){
                System.out.println("all label training finished");
                break;
            }
        }
        System.out.println("training done");
        File serializedModel =  new File(output,modelName);
        //todo pick best models

        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

        if (earlyStop){
            for (int l=0;l<numClasses;l++){
                System.out.println("----------------------------------------------------");
                System.out.println("test performance history for label "+l+": "+earlyStoppers.get(l).history());
                System.out.println("model size for label "+l+" = "+(boosting.getRegressors(l).size()-1));
            }
        }




    }

    static void tuneForMacroF(Config config) throws Exception{
        System.out.println("start tuning for macro F measure");
        String output = config.getString("output.folder");
        String modelName = "model_app3";
        double beta = config.getDouble("tune.FMeasure.beta");
        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        String tuneBy = config.getString("tune.data");
        String dataName;
        switch (tuneBy){
            case "train":
                dataName = config.getString("input.trainData");
                break;
            case "test":
                dataName = config.getString("input.testData");
                break;
            default:
                throw new IllegalArgumentException("tune.data should be train or test");
        }


        MultiLabelClfDataSet dataSet = loadData(config,dataName);
        double[] thresholds = MacroFMeasureTuner.tuneThresholds(boosting,dataSet,beta);
        TunedMarginalClassifier  tunedMarginalClassifier = new TunedMarginalClassifier(boosting,thresholds);
        Serialization.serialize(tunedMarginalClassifier, new File(output,"predictor_macro_f"));
        System.out.println("finish tuning for macro F measure");

    }

    static void report(Config config, String dataName) throws Exception{
        System.out.println("generating reports for data set "+dataName);
        String output = config.getString("output.folder");
        String modelName = "model_app3";
        File analysisFolder = new File(new File(output,"reports_app3"),dataName+"_reports");
        analysisFolder.mkdirs();
        FileUtils.cleanDirectory(analysisFolder);

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        String predictTarget = config.getString("predict.target");

        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = null;

        switch (predictTarget){
            case "subsetAccuracy":
                pluginPredictorTmp = new SubsetAccPredictor(boosting);
                break;
            case "hammingLoss":
                pluginPredictorTmp = new HammingPredictor(boosting);
                break;
            case "instanceFMeasure":
                pluginPredictorTmp = new InstanceF1Predictor(boosting);
                break;
            case "macroFMeasure":
                TunedMarginalClassifier  tunedMarginalClassifier = (TunedMarginalClassifier)Serialization.deserialize(new File(output, "predictor_macro_f"));
                pluginPredictorTmp = new MacroF1Predictor(boosting,tunedMarginalClassifier);
                break;
            default:
                throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
        }

        // just to make Lambda expressions happy
        final PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        MultiLabelClfDataSet dataSet = loadData(config,dataName);

        MLMeasures mlMeasures = new MLMeasures(pluginPredictor,dataSet);
        mlMeasures.getMacroAverage().setLabelTranslator(boosting.getLabelTranslator());

        System.out.println("performance on dataset "+dataName);
        System.out.println(mlMeasures);


        boolean simpleCSV = true;
        if (simpleCSV){
            System.out.println("start generating simple CSV report");
            double probThreshold=config.getDouble("report.classProbThreshold");
            File csv = new File(analysisFolder,"report.csv");
            List<String> strs = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToObj(i->IMLGBInspector.simplePredictionAnalysis(boosting,pluginPredictor,dataSet,i,probThreshold))
                    .collect(Collectors.toList());
            StringBuilder sb = new StringBuilder();
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                String str = strs.get(i);
                sb.append(str);

            }
            FileUtils.writeStringToFile(csv,sb.toString(),false);
            System.out.println("finish generating simple CSV report");
        }


        boolean topFeaturesToJson = false;
        File distributionFile = new File(new File(config.getString("input.folder"), "meta_data"),"distributions.ser");
        if (distributionFile.exists()){
            topFeaturesToJson = true;
        }
        if (topFeaturesToJson){
            System.out.println("start writing top features");
            Collection<FeatureDistribution> distributions = (Collection) Serialization.deserialize(distributionFile);
            int limit = config.getInt("report.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0,boosting.getNumClasses())
                    .mapToObj(k -> IMLGBInspector.topFeatures(boosting, k, limit, distributions))
                    .collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = "top_features.json";
            mapper.writeValue(new File(analysisFolder,file), topFeaturesList);
            System.out.println("finish writing top features");
        }


        boolean rulesToJson = true;
        if (rulesToJson){
            System.out.println("start writing rules to json");
            int ruleLimit = config.getInt("report.rule.limit");
            int numDocsPerFile = config.getInt("report.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)dataSet.getNumDataPoints()/numDocsPerFile);

            double probThreshold=config.getDouble("report.classProbThreshold");
            int labelSetLimit = config.getInt("report.labelSetLimit");


            IntStream.range(0,numFiles).forEach(i->{
                int start = i*numDocsPerFile;
                int end = start+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = IntStream.range(start,Math.min(end,dataSet.getNumDataPoints())).parallel().mapToObj(a->
                    IMLGBInspector.analyzePrediction(boosting, pluginPredictor, dataSet, a,  ruleLimit,labelSetLimit, probThreshold)).collect(Collectors.toList());
                ObjectMapper mapper = new ObjectMapper();

                String file = "report_"+(i+1)+".json";
                try {
                    mapper.writeValue(new File(analysisFolder,file), partition);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                System.out.println("progress = "+ Progress.percentage(i+1,numFiles));
            });

            System.out.println("finish writing rules to json");
        }


        boolean dataInfoToJson = true;
        if (dataInfoToJson){
            System.out.println("start writing data info to json");
            Set<String> modelLabels = IntStream.range(0,boosting.getNumClasses()).mapToObj(i->boosting.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            Set<String> dataSetLabels = DataSetUtil.gatherLabels(dataSet).stream().map(i -> dataSet.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            JsonGenerator jsonGenerator = new JsonFactory().createGenerator(new File(analysisFolder,"data_info.json"), JsonEncoding.UTF8);
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("dataSet",dataName);
            jsonGenerator.writeNumberField("numClassesInModel",boosting.getNumClasses());
            jsonGenerator.writeNumberField("numClassesInDataSet",dataSetLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelDataSetCombined",dataSet.getNumClasses());
            Set<String> modelNotDataLabels = SetUtil.complement(modelLabels, dataSetLabels);
            Set<String> dataNotModelLabels = SetUtil.complement(dataSetLabels,modelLabels);
            jsonGenerator.writeNumberField("numClassesInDataSetButNotModel",dataNotModelLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelButNotDataSet",modelNotDataLabels.size());
            jsonGenerator.writeArrayFieldStart("classesInDataSetButNotModel");
            for (String label: dataNotModelLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeArrayFieldStart("classesInModelButNotDataSet");
            for (String label: modelNotDataLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeNumberField("labelCardinality",dataSet.labelCardinality());

            jsonGenerator.writeEndObject();
            jsonGenerator.close();
            System.out.println("finish writing data info to json");
        }


        boolean modelConfigToJson = true;
        if (modelConfigToJson){
            System.out.println("start writing model config to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"model_config.json"),config);
            System.out.println("finish writing model config to json");
        }

        boolean dataConfigToJson = true;
        if (dataConfigToJson){
            System.out.println("start writing data config to json");
            File dataConfigFile = Paths.get(config.getString("input.folder"),
                    "data_sets",dataName,"data_config.json").toFile();
            if (dataConfigFile.exists()){
                FileUtils.copyFileToDirectory(dataConfigFile,analysisFolder);
            }
            System.out.println("finish writing data config to json");
        }

        boolean performanceToJson = true;
        if (performanceToJson){
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"performance.json"),mlMeasures);
        }

        boolean individualPerformance = true;
        if (individualPerformance){
            System.out.println("start writing individual label performance to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"individual_performance.json"),mlMeasures.getMacroAverage());
            System.out.println("finish writing individual label performance to json");
        }

        System.out.println("reports generated");
    }



    private static double KL(IMLGradientBoosting boosting, Vector vector, MultiLabel multiLabel, int classIndex){
        double[] p = new double[2];
        if (multiLabel.matchClass(classIndex)){
            p[0] = 0;
            p[1] = 1;
        } else {
            p[0] = 1;
            p[1] = 0;
        }
        double[] logQ = boosting.predictLogClassProbs(vector, classIndex);
        return KLDivergence.klGivenPLogQ(p, logQ);
    }

    private static double KL(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, int classIndex){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->KL(boosting, dataSet.getRow(i), dataSet.getMultiLabels()[i], classIndex))
                .average().getAsDouble();
    }


}
