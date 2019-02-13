package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.BRInspector;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.IndependentPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor;
import edu.neu.ccs.pyramid.util.*;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class BRPrediction {
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


        if (config.getBoolean("test")){
            test(config, logger);
        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    private static void test(Config config, Logger logger) throws Exception{
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

        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), dataSetType, true);
        MultiLabelClassifier.ClassProbEstimator classProbEstimator= (MultiLabelClassifier.ClassProbEstimator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","classifier"));
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
                classifier = (Reranker)setCalibrator;
                break;

            default:
                throw new IllegalArgumentException("illegal predict.mode");
        }
        MultiLabel[] predictions = classifier.predict(test);

        logger.info("test performance");
        logger.info(new MLMeasures(test.getNumClasses(),test.getMultiLabels(), predictions).toString());

        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);

        if (true) {
            logger.info("calibration performance on "+config.getString("input.testFolder")+" set");

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, test.getNumDataPoints()).parallel()
                    .boxed().map(i -> calibrationDataGenerator.createInstance(classProbEstimator, test.getRow(i),predictions[i],test.getMultiLabels()[i]))
                    .collect(Collectors.toList());

            BRCalibration.eval(instances, setCalibrator, logger);

            double confidenceThreshold = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                    "ctat",config.getString("CTAT.name")).toFile()));
            CTAT.Summary summary = CTAT.applyThreshold(BRCalibration.generateStream(instances,setCalibrator),confidenceThreshold);
            logger.info("autocoding performance on dataset "+config.getString("input.testFolder")+"  with unclipped confidence threshold "+summary.getConfidenceThreshold());
            logger.info("autocoding percentage = "+ summary.getAutoCodingPercentage());
            logger.info("autocoding accuracy = "+ summary.getAutoCodingAccuracy());
            logger.info("number of autocoded documents = "+ summary.getNumAutoCoded());
            logger.info("number of correct autocoded documents = "+ summary.getNumCorrectAutoCoded());

            double confidenceThresholdClipped = Double.parseDouble(FileUtils.readFileToString(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                    "ctat",config.getString("CTAT.name")+"_clipped").toFile()));
            CTAT.Summary summaryClipped = CTAT.applyThreshold(BRCalibration.generateStream(instances,setCalibrator),confidenceThresholdClipped);
            logger.info("autocoding performance on dataset "+config.getString("input.testFolder")+"  with clipped confidence threshold "+summaryClipped.getConfidenceThreshold());
            logger.info("autocoding percentage = "+ summaryClipped.getAutoCodingPercentage());
            logger.info("autocoding accuracy = "+ summaryClipped.getAutoCodingAccuracy());
            logger.info("number of autocoded documents = "+ summaryClipped.getNumAutoCoded());
            logger.info("number of correct autocoded documents = "+ summaryClipped.getNumCorrectAutoCoded());
        }


        MultiLabelClassifier fClassifier = classifier;

        boolean simpleCSV = true;
        if (simpleCSV){
            File testDataFile = new File(config.getString("input.testData"));
            File csv = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",testDataFile.getName()+"_reports","report.csv").toFile();
            csv.getParentFile().mkdirs();
            List<Integer> list = IntStream.range(0,test.getNumDataPoints()).boxed().collect(Collectors.toList());
            ParallelStringMapper<Integer> mapper = (list1, i) -> simplePredictionAnalysisCalibrated(classProbEstimator, labelCalibrator, setCalibrator,
                    test, i, fClassifier,predictionFeatureExtractor);
            ParallelFileWriter.mapToString(mapper,list, csv,100);
        }


        boolean topSets = true;
        if (topSets){
            File testDataFile = new File(config.getString("input.testData"));
            File csv = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",testDataFile.getName()+"_reports","top_sets.csv").toFile();
            csv.getParentFile().mkdirs();
            List<Integer> list = IntStream.range(0,test.getNumDataPoints()).boxed().collect(Collectors.toList());
            ParallelStringMapper<Integer> mapper = (list1, i) -> topKSets(config, classProbEstimator, labelCalibrator, setCalibrator,
                    test, i, fClassifier,predictionFeatureExtractor);
            ParallelFileWriter.mapToString(mapper,list, csv,100);
        }



        boolean rulesToJson = config.getBoolean("report.showPredictionDetail");
        if (rulesToJson){
            logger.info("start writing rules to json");
            int ruleLimit = config.getInt("report.rule.limit");
            int numDocsPerFile = config.getInt("report.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)test.getNumDataPoints()/numDocsPerFile);

            double probThreshold=config.getDouble("report.classProbThreshold");
            int labelSetLimit = config.getInt("report.labelSetLimit");


            IntStream.range(0,numFiles).forEach(i->{
                int start = i*numDocsPerFile;
                int end = start+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = IntStream.range(start,Math.min(end,test.getNumDataPoints())).parallel().mapToObj(a->
                        BRInspector.analyzePrediction(classProbEstimator, labelCalibrator, setCalibrator, test, fClassifier, predictionFeatureExtractor, a,  ruleLimit,labelSetLimit, probThreshold))
                        .collect(Collectors.toList());
                ObjectMapper mapper = new ObjectMapper();
                File testDataFile = new File(config.getString("input.testData"));
                File jsonFile = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",testDataFile.getName()+"_reports","report_"+(i+1)+".json").toFile();
                try {
                    mapper.writeValue(jsonFile, partition);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                logger.info("progress = "+ Progress.percentage(i+1,numFiles));
            });

            logger.info("finish writing rules to json");
        }

    }


    public static String simplePredictionAnalysisCalibrated(MultiLabelClassifier.ClassProbEstimator classProbEstimator,
                                                            LabelCalibrator labelCalibrator,
                                                            VectorCalibrator setCalibrator,
                                                            MultiLabelClfDataSet dataSet,
                                                            int dataPointIndex,
                                                            MultiLabelClassifier classifier,
                                                            PredictionFeatureExtractor predictionFeatureExtractor){
        StringBuilder sb = new StringBuilder();
        MultiLabel trueLabels = dataSet.getMultiLabels()[dataPointIndex];
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        double[] classProbs = classProbEstimator.predictClassProbs(dataSet.getRow(dataPointIndex));
        double[] calibratedClassProbs = labelCalibrator.calibratedClassProbs(classProbs);

        MultiLabel predicted = classifier.predict(dataSet.getRow(dataPointIndex));

        List<Integer> classes = new ArrayList<Integer>();
        for (int k = 0; k < dataSet.getNumClasses(); k++){
            if (dataSet.getMultiLabels()[dataPointIndex].matchClass(k)
                    ||predicted.matchClass(k)){
                classes.add(k);
            }
        }

        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<Integer,Double>> list = classes.stream().map(l -> {
            if (l < classProbEstimator.getNumClasses()) {
                return new Pair<>(l, calibratedClassProbs[l]);
            } else {
                return new Pair<>(l, 0.0);
            }
        }).sorted(comparator.reversed()).collect(Collectors.toList());
        for (Pair<Integer,Double> pair: list){
            int label = pair.getFirst();
            double prob = pair.getSecond();
            int match = 0;
            if (trueLabels.matchClass(label)){
                match=1;
            }
            sb.append(id).append("\t").append(labelTranslator.toExtLabel(label)).append("\t")
                    .append("single").append("\t").append(prob)
                    .append("\t").append(match).append("\n");
        }

        PredictionCandidate predictedCandidate = new PredictionCandidate();
        predictedCandidate.multiLabel = predicted;
        predictedCandidate.labelProbs = calibratedClassProbs;
        predictedCandidate.x = dataSet.getRow(dataPointIndex);
        Vector feature = predictionFeatureExtractor.extractFeatures(predictedCandidate);
        double probability = setCalibrator.calibrate(feature);


        List<Integer> predictedList = predicted.getMatchedLabelsOrdered();
        sb.append(id).append("\t");
        for (int i=0;i<predictedList.size();i++){
            sb.append(labelTranslator.toExtLabel(predictedList.get(i)));
            if (i!=predictedList.size()-1){
                sb.append(",");
            }
        }
        sb.append("\t");
        int setMatch = 0;
        if (predicted.equals(trueLabels)){
            setMatch=1;
        }
        sb.append("set").append("\t").append(probability).append("\t").append(setMatch).append("\n");
        return sb.toString();
    }


    private static String topKSets(Config config,
                                   MultiLabelClassifier.ClassProbEstimator classProbEstimator,
                                   LabelCalibrator labelCalibrator,
                                   VectorCalibrator setCalibrator,
                                   MultiLabelClfDataSet dataSet,
                                   int dataPointIndex, MultiLabelClassifier classifier, PredictionFeatureExtractor predictionVectorizer){
        StringBuilder sb = new StringBuilder();
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        int top = config.getInt("report.labelSetLimit");

        List<Pair<MultiLabel,Double>> topK;
        if (classifier instanceof SupportPredictor){
            topK = TopKFinder.topKinSupport(dataSet.getRow(dataPointIndex),classProbEstimator,labelCalibrator,setCalibrator,
                    predictionVectorizer,((SupportPredictor)classifier).getSupport(),top);
        } else {
            topK = TopKFinder.topK(dataSet.getRow(dataPointIndex),classProbEstimator,labelCalibrator,setCalibrator,
                    predictionVectorizer,top);
        }


        for (Pair<MultiLabel,Double> pair: topK){
            MultiLabel set = pair.getFirst();
            double probability = pair.getSecond();
            List<Integer> predictedList = set.getMatchedLabelsOrdered();
            sb.append(id).append("\t");
            for (int i=0;i<predictedList.size();i++){
                sb.append(labelTranslator.toExtLabel(predictedList.get(i)));
                if (i!=predictedList.size()-1){
                    sb.append(",");
                }
            }
            sb.append("\t");
            sb.append(probability);
            sb.append("\n");
        }

        return sb.toString();
    }








}
