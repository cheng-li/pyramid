package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.eval.Recall;
import edu.neu.ccs.pyramid.multilabel_classification.BRInspector;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.IndependentPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor;
import edu.neu.ccs.pyramid.util.*;
import edu.neu.ccs.pyramid.visualization.Visualizer;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
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


        if (config.getBoolean("validate")){
            report(config, config.getString("input.validData"), logger);
        }

        if (config.getBoolean("test")){
            report(config, config.getString("input.testData"), logger);
        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    public static void reportValid(Config config) throws Exception{
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


        if (config.getBoolean("validate")){
            report(config, config.getString("input.validData"), logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }


    public static void reportTest(Config config) throws Exception{
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
            report(config, config.getString("input.testData"), logger);
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    private static void report(Config config, String dataPath, Logger logger) throws Exception{
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

        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(dataPath, dataSetType, true);
        MultiLabelClassifier.ClassProbEstimator classProbEstimator= (MultiLabelClassifier.ClassProbEstimator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","classifier"));
        LabelCalibrator labelCalibrator = (LabelCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"label_calibrator").toFile());
        VectorCalibrator setCalibrator = (VectorCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"set_calibrator").toFile());
        PredictionFeatureExtractor predictionFeatureExtractor = (PredictionFeatureExtractor) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models",
                "calibrators",config.getString("output.calibratorFolder"),"prediction_feature_extractor").toFile());

        File testDataFile = new File(dataPath);

        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"models","support").toFile());

        String reportFolder = Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"predictions",testDataFile.getName()+"_reports").toString();

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
        MultiLabel[] predictions = classifier.predict(test);

        logger.info("classification performance on dataset "+testDataFile.getName());
        MLMeasures mlMeasures = new MLMeasures(test.getNumClasses(),test.getMultiLabels(), predictions);
        logger.info(mlMeasures.toString());

        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);

        if (true) {
            logger.info("calibration performance on dataset " + testDataFile.getName());

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, test.getNumDataPoints()).parallel()
                    .boxed().map(i -> calibrationDataGenerator.createInstance(classProbEstimator, test.getRow(i), predictions[i], test.getMultiLabels()[i]))
                    .collect(Collectors.toList());

            BRCalibration.eval(instances, setCalibrator, logger);

        }




        MultiLabelClassifier fClassifier = classifier;

        boolean simpleCSV = true;

        if (simpleCSV){

            File csv = Paths.get(reportFolder,"report.csv").toFile();
            csv.getParentFile().mkdirs();
            if(csv.exists()){
                csv.delete();
            }
            StringBuilder sb = new StringBuilder();
            sb.append("doc_id").append("\t").append("predictions").append("\t").append("prediction_type").append("\t")
                    .append("confidence").append("\t").append("truth").append("\t").append("ground_truth").append("\t")
                    .append("precision").append("\t").append("recall").append("\t").append("F1").append("\n");

            FileUtils.writeStringToFile(csv,sb.toString());
            List<Integer> list = IntStream.range(0,test.getNumDataPoints()).boxed().collect(Collectors.toList());

            ParallelStringMapper<Integer> mapper = (list1, i) -> simplePredictionAnalysisCalibrated(classProbEstimator, labelCalibrator, setCalibrator,
                    test, i, fClassifier,predictionFeatureExtractor);
            ParallelFileWriter.mapToString(mapper,list, csv,100,true);
        }


        boolean topSets = true;
        if (topSets){

            File csv = Paths.get(reportFolder,"top_sets.csv").toFile();
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
                File jsonFile = Paths.get(reportFolder,"report_"+(i+1)+".json").toFile();
                try {
                    mapper.writeValue(jsonFile, partition);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                logger.info("progress = "+ Progress.percentage(i+1,numFiles));
            });

            logger.info("finish writing rules to json");
        }

        boolean individualPerformance = true;
        if (individualPerformance){

            logger.info("start writing individual label performance to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(Paths.get(reportFolder,"individual_performance.json").toFile(),mlMeasures.getMacroAverage());
            logger.info("finish writing individual label performance to json");
        }

        boolean dataConfigToJson = true;
        if (dataConfigToJson){
            logger.info("start writing data config to json");
            File dataConfigFile = Paths.get(dataPath,"data_config.json").toFile();
            if (dataConfigFile.exists()){
                FileUtils.copyFileToDirectory(dataConfigFile,new File(reportFolder));
            }
            logger.info("finish writing data config to json");
        }

        boolean dataInfoToJson = true;
        if (dataInfoToJson){
            logger.info("start writing data info to json");
            Set<String> modelLabels = IntStream.range(0,classifier.getNumClasses()).mapToObj(i->classProbEstimator.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            Set<String> dataSetLabels = DataSetUtil.gatherLabels(test).stream().map(i -> test.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            JsonGenerator jsonGenerator = new JsonFactory().createGenerator(Paths.get(reportFolder,"data_info.json").toFile(), JsonEncoding.UTF8);
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("dataSet",testDataFile.getName());
            jsonGenerator.writeNumberField("numClassesInModel",classifier.getNumClasses());
            jsonGenerator.writeNumberField("numClassesInDataSet",dataSetLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelDataSetCombined",test.getNumClasses());
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
            jsonGenerator.writeNumberField("labelCardinality",test.labelCardinality());

            jsonGenerator.writeEndObject();
            jsonGenerator.close();
            logger.info("finish writing data info to json");
        }


        boolean performanceToJson = true;
        if (performanceToJson){
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(Paths.get(reportFolder,"performance.json").toFile(),mlMeasures);
        }


        if (config.getBoolean("report.produceHTML")){
            logger.info("start producing html files");

            Config savedApp1Config = new Config(Paths.get(config.getString("output.dir"), "meta_data","saved_config_app1").toFile());

            List<String> hosts = savedApp1Config.getStrings("index.hosts");
            List<Integer> ports = savedApp1Config.getIntegers("index.ports");

            //todo make it better
            if (savedApp1Config.getString("index.clientType").equals("node")){
                hosts = new ArrayList<>();
                for (int port: ports){
                    hosts.add("localhost");
                }
                //default setting
                hosts.add("localhost");
                ports.add(9200);
            }
            try (Visualizer visualizer = new Visualizer(logger, hosts, ports)){
                visualizer.produceHtml(new File(reportFolder));
                logger.info("finish producing html files");
            }


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
                    .append("\t").append(match).append("\t").append("NA").append("\t").append("NA").append("\t")
                    .append("NA").append("\t").append("NA").append("\n");
        }

        PredictionCandidate predictedCandidate = new PredictionCandidate();
        predictedCandidate.multiLabel = predicted;
        predictedCandidate.labelProbs = calibratedClassProbs;
        predictedCandidate.x = dataSet.getRow(dataPointIndex);
        Vector feature = predictionFeatureExtractor.extractFeatures(predictedCandidate);
        double probability = setCalibrator.calibrate(feature);


        List<Integer> predictedList = predicted.getMatchedLabelsOrdered();
        MultiLabel prediction = new MultiLabel();
        sb.append(id).append("\t");
        for (int i=0;i<predictedList.size();i++){
            prediction.addLabel(predictedList.get(i));
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

        List<Integer> truthList = trueLabels.getMatchedLabels().stream().sorted().collect(Collectors.toList());
        StringBuilder sbLabels = new StringBuilder();
        for(int i = 0; i < truthList.size(); i++){
            if(i!=truthList.size()-1){
                sbLabels.append(labelTranslator.toExtLabel(truthList.get(i))).append(",");

            }else{

                sbLabels.append(labelTranslator.toExtLabel(truthList.get(i)));
            }
        }
        double precision = Precision.precision(trueLabels,prediction);
        double recall = Recall.recall(trueLabels,prediction);
        double f1 = FMeasure.f1(precision,recall);


        sb.append("set").append("\t").append(probability).append("\t").append(setMatch).append("\t")
                .append(sbLabels.toString()).append("\t").append(precision).append("\t").append(recall).append("\t")
                .append(f1).append("\n");
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
        int minSize = config.getInt("predict.minSize");
        int maxSize = config.getInt("predict.maxSize");

        List<Pair<MultiLabel,Double>> topK;
        if (classifier instanceof SupportPredictor){
            topK = TopKFinder.topKinSupport(dataSet.getRow(dataPointIndex),classProbEstimator,labelCalibrator,setCalibrator,
                    predictionVectorizer,((SupportPredictor)classifier).getSupport(),top);
        } else {
            topK = TopKFinder.topK(dataSet.getRow(dataPointIndex),classProbEstimator,labelCalibrator,setCalibrator,
                    predictionVectorizer,minSize, maxSize, top);
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
