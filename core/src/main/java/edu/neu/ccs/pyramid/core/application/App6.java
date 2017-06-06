package edu.neu.ccs.pyramid.core.application;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.*;
import edu.neu.ccs.pyramid.core.eval.MLMeasures;
import edu.neu.ccs.pyramid.core.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.core.multilabel_classification.crf.*;
import edu.neu.ccs.pyramid.core.util.PrintUtil;
import edu.neu.ccs.pyramid.core.util.Serialization;
import edu.neu.ccs.pyramid.core.util.SetUtil;
import edu.neu.ccs.pyramid.core.optimization.LBFGS;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * pair-wise CRF for multi-label classification
 * Created by chengli on 8/19/16.
 */
public class App6 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);

        }

        if (config.getBoolean("test")){
            test(config);
        }

    }

    private static void train(Config config) throws Exception{
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);

        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(trainSet);
        double gaussianVariance = config.getDouble("train.gaussianVariance");
        cmlcrf.setConsiderPair(true);
        CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);

        int maxIteration = config.getInt("train.maxIteration");
        crfLoss.setRegularizeAll(true);
        LBFGS optimizer = new LBFGS(crfLoss);
        optimizer.getTerminator().setMaxIteration(maxIteration);

        PluginPredictor<CMLCRF> predictor = null;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                predictor = new SubsetAccPredictor(cmlcrf);
                break;
            case "instanceFMeasure":
                predictor = new InstanceF1Predictor(cmlcrf);
                break;
            default:
                throw new IllegalArgumentException("predict.target must be subsetAccuracy or instanceFMeasure");
        }

        int progressInterval = config.getInt("train.showProgress.interval");
        System.out.println("start training");
        int iteration = 0;
        while(true){
            optimizer.iterate();
            iteration += 1;

            if (iteration%progressInterval==0){
                System.out.println("iteration "+iteration);
                System.out.println("training objective = "+optimizer.getTerminator().getLastValue());
                System.out.println("training performance:");
                System.out.println(new MLMeasures(predictor,trainSet));
                System.out.println("test performance:");
                System.out.println(new MLMeasures(predictor,testSet));
            }

            if (optimizer.getTerminator().shouldTerminate()){
                System.out.println("iteration "+iteration);
                System.out.println("training objective = "+optimizer.getTerminator().getLastValue());
                System.out.println("training performance:");
                System.out.println(new MLMeasures(predictor,trainSet));
                System.out.println("test performance:");
                System.out.println(new MLMeasures(predictor,testSet));
                System.out.println("training done!");
                break;
            }
        }

        String modelName = "model_crf";
        String output = config.getString("output.folder");
        (new File(output)).mkdirs();
        File serializeModel = new File(output, modelName);
        cmlcrf.serialize(serializeModel);

        MultiLabel[] predictions = cmlcrf.predict(trainSet);
        File predictionFile = new File(output, "train_predictions.txt");
        FileUtils.writeStringToFile(predictionFile, PrintUtil.toMutipleLines(predictions));
        System.out.println("predictions on the training set are written to "+predictionFile.getAbsolutePath());

        if (config.getBoolean("train.generateReports")){
            report(config, trainSet, "trainSet");
        }
    }


    private static void test(Config config) throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        String modelName = "model_crf";
        String output = config.getString("output.folder");
        CMLCRF cmlcrf = (CMLCRF) Serialization.deserialize(new File(output, modelName));
        PluginPredictor<CMLCRF> predictor = null;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                predictor = new SubsetAccPredictor(cmlcrf);
                break;
            case "instanceFMeasure":
                predictor = new InstanceF1Predictor(cmlcrf);
                break;
            default:
                throw new IllegalArgumentException("predict.target must be subsetAccuracy or instanceFMeasure");
        }
        System.out.println("test performance:");
        System.out.println(new MLMeasures(predictor,testSet));

        MultiLabel[] predictions = cmlcrf.predict(testSet);
        File predictionFile = new File(output, "test_predictions.txt");
        FileUtils.writeStringToFile(predictionFile, PrintUtil.toMutipleLines(predictions));
        System.out.println("predictions on the test set are written to "+predictionFile.getAbsolutePath());

        report(config, testSet, "testSet");

    }

    static void report(Config config, MultiLabelClfDataSet dataSet, String dataName) throws Exception{
        System.out.println("generating reports for data set "+dataName);
        String output = config.getString("output.folder");
        String modelName = "model_crf";
        File analysisFolder = new File(new File(output,"reports_crf"),dataName+"_reports");
        analysisFolder.mkdirs();
        FileUtils.cleanDirectory(analysisFolder);

        CMLCRF crf = (CMLCRF)Serialization.deserialize(new File(output, modelName));
        PluginPredictor<CMLCRF> predictorTmp = null;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                predictorTmp = new SubsetAccPredictor(crf);
                break;
            case "instanceFMeasure":
                predictorTmp = new InstanceF1Predictor(crf);
                break;
            default:
                throw new IllegalArgumentException("predict.target must be subsetAccuracy or instanceFMeasure");
        }

        // just to make Lambda expressions happy
        final PluginPredictor<CMLCRF> predictor = predictorTmp;



        MLMeasures mlMeasures = new MLMeasures(predictor,dataSet);
        mlMeasures.getMacroAverage().setLabelTranslator(crf.getLabelTranslator());

        System.out.println("performance on dataset "+dataName);
        System.out.println(mlMeasures);


        boolean simpleCSV = true;
        if (simpleCSV){
//            System.out.println("start generating simple CSV report");
            double probThreshold=config.getDouble("report.classProbThreshold");
            File csv = new File(analysisFolder,"report.csv");
            List<String> strs = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToObj(i-> CRFInspector.simplePredictionAnalysis(crf,predictor,dataSet,i,probThreshold))
                    .collect(Collectors.toList());
            StringBuilder sb = new StringBuilder();
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                String str = strs.get(i);
                sb.append(str);

            }
            FileUtils.writeStringToFile(csv,sb.toString(),false);
//            System.out.println("finish generating simple CSV report");
        }




        boolean dataInfoToJson = true;
        if (dataInfoToJson){
//            System.out.println("start writing data info to json");
            Set<String> modelLabels = IntStream.range(0,crf.getNumClasses()).mapToObj(i->crf.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            Set<String> dataSetLabels = DataSetUtil.gatherLabels(dataSet).stream().map(i -> dataSet.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            JsonGenerator jsonGenerator = new JsonFactory().createGenerator(new File(analysisFolder,"data_info.json"), JsonEncoding.UTF8);
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("dataSet",dataName);
            jsonGenerator.writeNumberField("numClassesInModel",crf.getNumClasses());
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
//            System.out.println("finish writing data info to json");
        }


        boolean modelConfigToJson = true;
        if (modelConfigToJson){
//            System.out.println("start writing model config to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"model_config.json"),config);
//            System.out.println("finish writing model config to json");
        }
        

        boolean performanceToJson = true;
        if (performanceToJson){
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"performance.json"),mlMeasures);
        }

        boolean individualPerformance = true;
        if (individualPerformance){
//            System.out.println("start writing individual label performance to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"individual_performance.json"),mlMeasures.getMacroAverage());
//            System.out.println("finish writing individual label performance to json");
        }

        System.out.println("reports generated");
    }
}
