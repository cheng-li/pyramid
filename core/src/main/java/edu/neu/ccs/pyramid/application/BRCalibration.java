package edu.neu.ccs.pyramid.application;


import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.CalibrationEval;
import edu.neu.ccs.pyramid.eval.MLMeasures;

import edu.neu.ccs.pyramid.feature.CategoricalFeature;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import edu.neu.ccs.pyramid.multilabel_classification.predictor.IndependentPredictor;

import edu.neu.ccs.pyramid.util.*;
import org.apache.commons.io.FileUtils;


import java.io.File;

import java.io.Serializable;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BRCalibration {
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

        if (config.getBoolean("calibrate")){
            calibrate(config, logger);
        }

        if(config.getBoolean("tuneCTAT")){
            tuneCTAT(config,logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        main(config);

    }


    private static void calibrate(Config config, Logger logger) throws Exception{


        logger.info("start training calibrators");

        DataSetType dataSetType;
        switch (config.getString("dataSetType")){
            case "sparse_random":
                dataSetType = DataSetType.ML_CLF_SPARSE;
                break;
            case "sparse_sequential":
                dataSetType = DataSetType.ML_CLF_SEQ_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("unknown dataSetType");
        }

        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), dataSetType, true);

        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.calibrationData"), dataSetType, true);

        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validData"), dataSetType, true);

        MultiLabelClassifier.ClassProbEstimator classProbEstimator = (MultiLabelClassifier.ClassProbEstimator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","classifier"));

        List<Integer> labelCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==0).boxed().collect(Collectors.toList());
        List<Integer> setCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==1).boxed().collect(Collectors.toList());

        MultiLabelClfDataSet labelCalData = DataSetUtil.sampleData(cal, labelCalIndices);
        MultiLabelClfDataSet setCalData = DataSetUtil.sampleData(cal, setCalIndices);

        logger.info("start training label calibrator");
        LabelCalibrator labelCalibrator = null;
        switch (config.getString("labelCalibrator")){
            case "isotonic":
                labelCalibrator = new IsoLabelCalibrator(classProbEstimator, labelCalData);
                break;
            case "none":
                labelCalibrator = new IdentityLabelCalibrator();
                break;
        }
        logger.info("finish training label calibrator");

        logger.info("start training set calibrator");

        List<PredictionFeatureExtractor> extractors = new ArrayList<>();

        if (config.getBoolean("brProb")){
            extractors.add(new BRProbFeatureExtractor());
        }

        if (config.getBoolean("setPrior")){
            extractors.add(new PriorFeatureExtractor(train));
        }

        if (config.getBoolean("card")){
            extractors.add(new CardFeatureExtractor());
        }

        if (config.getBoolean("encodeLabel")){
            extractors.add(new LabelBinaryFeatureExtractor(classProbEstimator.getNumClasses(),train.getLabelTranslator()));
        }

        if (config.getBoolean("useInitialFeatures")){
            Set<String> prefixes = new HashSet<>(config.getStrings("featureFieldPrefix"));
            FeatureList featureList = train.getFeatureList();
            List<Integer> featureIds = new ArrayList<>();
            for (int j=0;j<featureList.size();j++){
                Feature feature = featureList.get(j);
                if (feature instanceof CategoricalFeature){
                    if (matchPrefixes(((CategoricalFeature) feature).getVariableName(),prefixes)){
                        featureIds.add(j);
                    }
                } else {
                    if ( !(feature instanceof Ngram)){
                        if (matchPrefixes(feature.getName(),prefixes)){
                            featureIds.add(j);
                        }
                    }
                }
            }

            extractors.add(new InstanceFeatureExtractor(featureIds,train.getFeatureList()));
        }


        PredictionFeatureExtractor predictionFeatureExtractor = new CombinedPredictionFeatureExtractor(extractors);

        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);
        CalibrationDataGenerator.TrainData caliTrainingData = calibrationDataGenerator.createCaliTrainingData(setCalData,classProbEstimator,config.getInt("numCandidates"));

        CalibrationDataGenerator.TrainData caliValidData = calibrationDataGenerator.createCaliTrainingData(valid,classProbEstimator,config.getInt("numCandidates"));


        RegDataSet calibratorTrainData = caliTrainingData.regDataSet;
        double[] weights = caliTrainingData.instanceWeights;

        VectorCalibrator setCalibrator = null;

        switch (config.getString("setCalibrator")){
            case "cardinality_isotonic":
                setCalibrator = new VectorCardIsoSetCalibrator(calibratorTrainData, 0, 2);
                break;
            case "reranker":
                RerankerTrainer rerankerTrainer = RerankerTrainer.newBuilder()
                        .numCandidates(config.getInt("numCandidates"))
                        .monotonic(config.getBoolean("monotonic"))
                        .numLeaves(config.getInt("numLeaves"))
                        .strongMonotonicity(false)
                        .build();
                setCalibrator = rerankerTrainer.trainWithSigmoid(calibratorTrainData, weights,classProbEstimator,predictionFeatureExtractor,labelCalibrator, caliValidData.regDataSet);
                break;
            case "isotonic":
                setCalibrator = new VectorIsoSetCalibrator(calibratorTrainData,0);
                break;
            case "none":
                setCalibrator = new VectorIdentityCalibrator(0);
                break;
            default:
                throw new IllegalArgumentException("illegal setCalibrator");
        }

        logger.info("finish training set calibrator");

        Serialization.serialize(labelCalibrator,Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"label_calibrator").toFile());
        Serialization.serialize(setCalibrator,Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"set_calibrator").toFile());
        Serialization.serialize(predictionFeatureExtractor,Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"prediction_feature_extractor").toFile());
        logger.info("finish training calibrators");



        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","support").toFile());



        MultiLabelClassifier classifier = null;
        switch (config.getString("predict.mode")){
            case "independent":
                classifier = new IndependentPredictor(classProbEstimator,labelCalibrator);
                break;
            case "support":
                classifier = new edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor(classProbEstimator, labelCalibrator, support);
                break;

            case "reranker":
                Reranker reranker = (Reranker)setCalibrator;
                reranker.setMinPredictionSize(config.getInt("predict.minSize"));
                reranker.setMaxPredictionSize(config.getInt("predict.maxSize"));
                classifier = reranker;
                break;

            default:
                throw new IllegalArgumentException("illegal predict.mode");
        }
        MultiLabel[] predictions = classifier.predict(cal);

        MultiLabel[] predictions_valid = classifier.predict(valid);


        if (true) {
            logger.info("calibration performance on "+config.getString("input.calibrationFolder")+ " set");

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, cal.getNumDataPoints()).parallel()
                    .boxed().map(i -> calibrationDataGenerator.createInstance(classProbEstimator, cal.getRow(i),predictions[i],cal.getMultiLabels()[i]))
                    .collect(Collectors.toList());

            eval(instances, setCalibrator, logger);
        }

        logger.info("classification performance on "+config.getString("input.validFolder")+" set");
        logger.info(new MLMeasures(valid.getNumClasses(),valid.getMultiLabels(), predictions_valid).toString());

        if (true) {
            logger.info("calibration performance on "+ config.getString("input.validFolder")+" set");

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, valid.getNumDataPoints()).parallel()
                    .boxed().map(i -> calibrationDataGenerator.createInstance(classProbEstimator, valid.getRow(i),predictions_valid[i],valid.getMultiLabels()[i]))
                    .collect(Collectors.toList());

            eval(instances, setCalibrator, logger);

        }



    }


    private static boolean matchPrefixes(String name, Set<String> prefixes){
        for (String prefix: prefixes){
            if (name.startsWith(prefix)){
                return true;
            }
        }
        return false;
    }

    private static void tuneCTAT(Config config, Logger logger)throws Exception{
        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validData"), DataSetType.ML_CLF_SEQ_SPARSE,true);
        MultiLabelClassifier.ClassProbEstimator classProbEstimator = (MultiLabelClassifier.ClassProbEstimator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","classifier"));
        LabelCalibrator labelCalibrator = (LabelCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"label_calibrator").toFile());
        VectorCalibrator setCalibrator = (VectorCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"set_calibrator").toFile());
        PredictionFeatureExtractor predictionFeatureExtractor = (PredictionFeatureExtractor) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"prediction_feature_extractor").toFile());

        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","support").toFile());



        MultiLabelClassifier classifier = null;
        switch (config.getString("predict.mode")){
            case "independent":
                classifier = new IndependentPredictor(classProbEstimator,labelCalibrator);
                break;
            case "support":
                classifier = new edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor(classProbEstimator, labelCalibrator, support);
                break;

            case "reranker":
                Reranker reranker = (Reranker)setCalibrator;
                reranker.setMinPredictionSize(config.getInt("predict.minSize"));
                reranker.setMaxPredictionSize(config.getInt("predict.maxSize"));
                classifier = reranker;
                break;

            default:
                throw new IllegalArgumentException("illegal predict.mode");
        }

        MultiLabel[] predictions_valid = classifier.predict(valid);

        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);

        List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, valid.getNumDataPoints()).parallel()
                .boxed().map(i -> calibrationDataGenerator.createInstance(classProbEstimator, valid.getRow(i),predictions_valid[i],valid.getMultiLabels()[i]))
                .collect(Collectors.toList());
        double targetAccuracy = config.getDouble("CTAT.targetAccuracy");
        double confidenceThreshold = CTAT.findThreshold(generateStream(instances,setCalibrator),targetAccuracy).getConfidenceThreshold();


        logger.info("confidence threshold for target accuracy "+targetAccuracy +" = " +confidenceThreshold);

        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "ctat",config.getString("CTAT.name")).toFile(),""+confidenceThreshold);

        double confidenceThresholdClipped = CTAT.clip(confidenceThreshold,config.getDouble("CTAT.lowerBound"),config.getDouble("CTAT.upperBound"));

        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "ctat",config.getString("CTAT.name")+"_clipped").toFile(),""+confidenceThresholdClipped);

    }




    public static Stream<Pair<Double,Integer>> generateStream(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator vectorCalibrator){
        return predictions.stream()
                .parallel().map(pred->new Pair<>(vectorCalibrator.calibrate(pred.vector),(int)pred.correctness));
    }

    public static BRCalibration.CaliRes eval(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator calibrator, Logger logger){
        double mse = CalibrationEval.mse(generateStream(predictions,calibrator));
        double ace = CalibrationEval.absoluteError(generateStream(predictions,calibrator),10);
        double sharpness = CalibrationEval.sharpness(generateStream(predictions,calibrator),10);
        logger.info("mse="+mse);
        logger.info("absolute calibration error="+ace);
        logger.info("square calibration error="+CalibrationEval.squareError(generateStream(predictions,calibrator),10));
        logger.info("sharpness="+sharpness);
        logger.info("variance="+CalibrationEval.variance(generateStream(predictions,calibrator)));
        logger.info(Displayer.displayCalibrationResult(generateStream(predictions,calibrator)));

        BRCalibration.CaliRes caliRes = new BRCalibration.CaliRes();
        caliRes.mse = mse;
        caliRes.ace= ace;
        caliRes.sharpness = sharpness;
        return caliRes;
    }


    public static class CaliRes implements Serializable {
        public static final long serialVersionUID = 446782166720638575L;
        public double mse;
        public double ace;
        public double sharpness;
    }
}
