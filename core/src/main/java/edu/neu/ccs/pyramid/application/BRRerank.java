package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.CalibrationEval;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.IndependentPredictor;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BRRerank {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        if (config.getBoolean("trainBR")){
            System.out.println("Start training BR classifier");
            Config brConfig = produceBRConfig(config);
            CBMEN.main(brConfig);
            System.out.println("Finish training BR classifier");
        }


        Config caliConfig  = produceCaliConfig(config);
        calibrate(caliConfig);
        report(config);
        calibration_eval(config);
        classification_eval(config);

    }


    public static void calibrate(Config config) throws Exception{

        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("train"), DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("valid"),DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("cal"),DataSetType.ML_CLF_SPARSE,true);

        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("test"),DataSetType.ML_CLF_SPARSE,true);
        CBM cbm = (CBM) Serialization.deserialize(config.getString("cbm"));
        cbm.setAllowEmpty(config.getBoolean("allowEmpty"));

        MultiLabelClfDataSet labelCalData = cal;

        MultiLabelClfDataSet setCalData = cal;

        if (config.getBoolean("splitCalibrationData")){
            List<Integer> labelCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==0).boxed().collect(Collectors.toList());
            List<Integer> setCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==1).boxed().collect(Collectors.toList());
            labelCalData = DataSetUtil.sampleData(cal, labelCalIndices);
            setCalData = DataSetUtil.sampleData(cal, setCalIndices);
        }

        LabelCalibrator labelCalibrator = null;
        switch (config.getString("labelCalibrator")){
            case "isotonic":
                IsoLabelCalibrator isoLabelCalibrator = new IsoLabelCalibrator(cbm, labelCalData);
                isoLabelCalibrator.setConfidenceLowerBound(0);
                isoLabelCalibrator.setConfidenceUpperBound(1);
                labelCalibrator = isoLabelCalibrator;
                break;
            case "none":
                labelCalibrator = new IdentityLabelCalibrator();
                break;
        }

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
            extractors.add(new LabelBinaryFeatureExtractor(cbm.getNumClasses(),train.getLabelTranslator()));
        }


        PredictionFeatureExtractor predictionFeatureExtractor = new CombinedPredictionFeatureExtractor(extractors);

        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(train);

        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);
        CalibrationDataGenerator.TrainData caliTrainingData = calibrationDataGenerator.createCaliTrainingData(setCalData,cbm,config.getInt("numTrainCandidates"),"accuracy",support, 0);

        CalibrationDataGenerator.TrainData caliValidData = calibrationDataGenerator.createCaliTrainingData(valid,cbm,config.getInt("numTrainCandidates"),"accuracy",support,0);

        RegDataSet calibratorTrainData = caliTrainingData.regDataSet;
        double[] weights = caliTrainingData.instanceWeights;

        VectorCalibrator setCalibrator = null;

        switch (config.getString("setCalibrator")){
            case "trivial":
                setCalibrator = new VectorTrivialCalibrator(calibratorTrainData);
                break;
            case "cardinality_isotonic":
                setCalibrator = new VectorCardIsoSetCalibrator(calibratorTrainData, 0, 2,false);
                break;
            case "GB":
                RerankerTrainer rerankerTrainer = RerankerTrainer.newBuilder()
                        .numCandidates(config.getInt("numPredictCandidates"))
                        .numLeaves(config.getInt("numLeaves"))
                        .shrinkage(config.getDouble("shrinkage"))
                        .monotonicityType(config.getString("monotonicityType"))
                        .maxIter(config.getInt("maxIteration"))
                        .build();
                if (config.getString("trainingObjective").equals("MSE")){
                    setCalibrator = rerankerTrainer.train(calibratorTrainData, weights,cbm,predictionFeatureExtractor, labelCalibrator, caliValidData.regDataSet);
                }

                if (config.getString("trainingObjective").equals("KL")){
                    setCalibrator = rerankerTrainer.trainWithSigmoid(calibratorTrainData, weights,cbm,predictionFeatureExtractor, labelCalibrator, caliValidData.regDataSet);
                }
                break;

            case "isotonic":
                setCalibrator = new VectorIsoSetCalibrator(calibratorTrainData,0,false);
                break;
            case "none":
                setCalibrator = new VectorIdentityCalibrator(0);
                break;
            default:
                throw new IllegalArgumentException("illegal setCalibrator");
        }

        Serialization.serialize(labelCalibrator,Paths.get(config.getString("output"),"models","label_calibrator"));
        Serialization.serialize(setCalibrator,Paths.get(config.getString("output"),"models","set_calibrator"));
        Serialization.serialize(predictionFeatureExtractor,Paths.get(config.getString("output"),"models","calibration_feature_extractor"));
    }

    private static void report(Config config) throws Exception{
        MultiLabelClfDataSet dataset = TRECFormat.loadMultiLabelClfDataSet(Paths.get(config.getString("dataPath"),"test").toFile(),DataSetType.ML_CLF_SPARSE,true);
        CBM cbm = (CBM) Serialization.deserialize(Paths.get(config.getString("outputDir"),"models","model"));
        cbm.setAllowEmpty(true);
        MultiLabelClassifier classifier = null;

        LabelCalibrator labelCalibrator = (LabelCalibrator) Serialization.deserialize(Paths.get(config.getString("outputDir"),"models","label_calibrator"));
        VectorCalibrator setCalibrator = (VectorCalibrator) Serialization.deserialize(Paths.get(config.getString("outputDir"),"models","set_calibrator"));
        PredictionFeatureExtractor predictionFeatureExtractor = (PredictionFeatureExtractor) Serialization.deserialize(Paths.get(config.getString("outputDir"),"models","calibration_feature_extractor"));
        CalibrationDataGenerator calibrationDataGenerator = new CalibrationDataGenerator(labelCalibrator,predictionFeatureExtractor);
        switch (config.getString("predictMode")){

            case "independent":
                classifier = new IndependentPredictor(cbm,labelCalibrator);
                break;
            case "rerank":
                classifier = (Reranker)setCalibrator;
                break;

            default:
                throw new IllegalArgumentException("illegal predict.mode");
        }
        MultiLabel[] predictions = classifier.predict(dataset);


        List<CalibInfo> confidenceScores = IntStream.range(0, dataset.getNumDataPoints()).parallel()
                .boxed().map(i -> {CalibrationDataGenerator.CalibrationInstance predictionInstance = calibrationDataGenerator.createInstance(cbm, dataset.getRow(i),predictions[i],dataset.getMultiLabels()[i],"accuracy");
                    double calibrated = setCalibrator.calibrate(predictionInstance.vector);
                    CalibInfo calibInfo = new CalibInfo();
                    calibInfo.uncalibrated = predictionInstance.vector.get(0);
                    calibInfo.calibrated = calibrated;
                    calibInfo.accuracy = predictionInstance.correctness;
                    return calibInfo; }
                    ).collect(Collectors.toList());
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("set_prediction").append("\t").append("uncalibrated_confidence").append("\t").append("calibrated_confidence").append("\t").append("ground_truth").append("\t").append("set_accuracy").append("\n");
        for (int i=0;i<dataset.getNumDataPoints();i++){
            stringBuilder.append(predictions[i]).append("\t")
                    .append(confidenceScores.get(i).uncalibrated).append("\t")
                    .append(confidenceScores.get(i).calibrated).append("\t")
                    .append(dataset.getMultiLabels()[i]).append("\t")
                    .append(predictions[i].equals(dataset.getMultiLabels()[i])?1:0).append("\n");
        }

        FileUtils.writeStringToFile(Paths.get(config.getString("outputDir"),"reports","set_prediction_and_confidence.txt").toFile(),stringBuilder.toString());

    }

    private static void classification_eval(Config config) throws Exception{
        System.out.println("classification performance on test set");
        MultiLabelClfDataSet dataset = TRECFormat.loadMultiLabelClfDataSet(Paths.get(config.getString("dataPath"),"test").toFile(),DataSetType.ML_CLF_SPARSE,true);
        MultiLabel[] predictions = FileUtils.readLines(Paths.get(config.getString("outputDir"),"reports","set_prediction_and_confidence.txt").toFile())
                .stream().skip(1).map(line->new MultiLabel(line.split("\t")[0].replace("{","").replace("}",""),dataset.getLabelTranslator()))
                .toArray(MultiLabel[]::new);

        MLMeasures mlMeasures =new MLMeasures(dataset.getNumClasses(),dataset.getMultiLabels(), predictions);
        System.out.println("classification performance");
        System.out.println(mlMeasures);
        Paths.get(config.getString("outputDir")).toFile().mkdirs();
        FileUtils.writeStringToFile(Paths.get(config.getString("outputDir"),"reports","classification_performance.txt").toFile(),mlMeasures.toString());

    }

    private static void calibration_eval(Config config) throws Exception{
        List<Pair<Double,Double>> confidenceVsAccuracy= FileUtils.readLines(Paths.get(config.getString("outputDir"),"reports","set_prediction_and_confidence.txt").toFile())
                .stream().skip(1).map(line->{
                    double conf = Double.parseDouble(line.split("\t")[2]);
                    double acc = Double.parseDouble(line.split("\t")[4]);
                    Pair<Double,Double> pair = new Pair<>(conf,acc);
                    return pair;
                }).collect(Collectors.toList());
        double mse = CalibrationEval.mse(confidenceVsAccuracy.stream());
        double sharpness = CalibrationEval.sharpness(confidenceVsAccuracy.stream(),10);
        double alignment = CalibrationEval.squareError(confidenceVsAccuracy.stream(),10);
        double uncertainty=CalibrationEval.variance(confidenceVsAccuracy.stream());
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("mse=").append(mse).append("\n");
        stringBuilder.append("alignment error=").append(alignment).append("\n");
        stringBuilder.append("sharpness=").append(sharpness).append("\n");
        stringBuilder.append("uncertainty=").append(uncertainty).append("\n");

        System.out.println("calibration performance on test set");
        System.out.println(stringBuilder.toString());

        FileUtils.writeStringToFile(Paths.get(config.getString("outputDir"),"reports","calibration_performance.txt").toFile(),stringBuilder.toString());
    }



    private static Config produceBRConfig(Config config){
        Config br = getBRDefaultConfig();

        br.setString("input.trainData", Paths.get(config.getString("dataPath"),"train").toString());
        br.setString("input.testData", Paths.get(config.getString("dataPath"),"test").toString());
        br.setString("output.dir",Paths.get(config.getString("outputDir"),"models").toString());
        br.setString("train.iterations",config.getString("BR.iteration"));
        br.setString("train.penalty",config.getString("BR.penalty"));
        br.setString("train.l1Ratio",config.getString("BR.l1Ratio"));
        return br;
    }


    private static Config produceCaliConfig(Config config){
        Config cali = getCaliDefaultConfig();

        cali.setString("train", Paths.get(config.getString("dataPath"),"train").toString());
        cali.setString("cal", Paths.get(config.getString("dataPath"),"cal").toString());
        cali.setString("valid", Paths.get(config.getString("dataPath"),"valid").toString());
        cali.setString("test", Paths.get(config.getString("dataPath"),"test").toString());
        cali.setString("cbm",Paths.get(config.getString("outputDir"),"models","model").toString());
        cali.setString("output",config.getString("outputDir"));
        cali.setString("predict.mode",config.getString("predictMode"));
        String[] toCopy={"setPrior","brProb","card","encodeLabel","numTrainCandidates",
                "numPredictCandidates","shrinkage","numLeaves","labelCalibrator",
                "setCalibrator","minDataPerLeaf","trainingObjective","maxIteration"};
        Config.copy(config,cali,toCopy);
        return cali;

    }

    private static Config getCaliDefaultConfig(){
        String de = "monotonicityType=none\n" +
                "splitCalibrationData=true\n" +
                "allowEmpty=true";
        return Config.newConfigFromString(de);

    }

    private static Config getBRDefaultConfig(){
        String de = "############## input and output ###############\n" +

                "# Full path to input validation dataset, if available\n" +
                "# Used for hyper parameter tuning and early stopping\n" +
                "# If no additional validation set is available, leave it as blank,\n" +
                "# and random 20% of the training data will be used as the validation set.\n" +
                "input.validData=\n" +
                "\n" +
                "# Directory for the program output\n" +
                "\n" +
                "# Whether to show detailed debugging information\n" +
                "output.verbose=true\n" +
                "\n" +
                "################# functions #####################\n" +
                "\n" +
                "# Perform hyper parameter tuning before training\n" +
                "# If external validation data is given, the model is trained on the full training data\n" +
                "# and tuned on the given validation data; otherwise, the model is trained on 80% of the training data,\n" +
                "# and tuned on the rest 20% of the training data.\n" +
                "tune=false\n" +
                "\n" +
                "# Train the model on all the available data (excluding test data), using tuned or user specified hyper parameters\n" +
                "# If the external validation data is also given, the model is trained on training data + validation data\n" +
                "train=true\n" +
                "\n" +
                "# Load back trained model, make predictions on the test set, and evaluate test performance.\n" +
                "# The program shows several different predictions designed to optimize different evaluation metrics.\n" +
                "test=false\n" +
                "\n" +
                "######### prediction method ########\n" +
                "\n" +
                "# Whether to allow empty subset to be predicted;\n" +
                "# true = allow empty prediction\n" +
                "# false = do not allow empty prediction\n" +
                "# auto = allow empty prediction only if the training set contains empty label sets\n" +
                "predict.allowEmpty=true\n" +
                "\n" +
                "# The threshold for skipping components with small contributions\n" +
                "# This is designed to speed up prediction\n" +
                "predict.piThreshold=0.001\n" +
                "\n" +
                "\n" +
                "######### tune #########\n" +
                "\n" +
                "# Hyper parameter tuning uses the validation set to decide elasticnet penalty and L1 ratio,\n" +
                "# number of CBM components and number of EM training iterations.\n" +
                "# Users can specify candidate values for penalty, L1 ratio and components.\n" +
                "# The optimal EM training iterations will be determined automatically by monitoring the validation performance.\n" +
                "# The metric monitored is specified in predict.targetMetric.\n" +
                "\n" +
                "# To achieve optimal prediction under which target evaluation metric?\n" +
                "# Currently supported metrics: instance_set_accuracy, instance_f1 and instance_hamming_loss.\n" +
                "# Generally speaking no single model is well suited for all evaluation metrics.\n" +
                "# Optimizing different metrics require different prediction methods and hyper parameters.\n" +
                "# The program automatically chooses the optimal prediction method designed for each metric.\n" +
                "# The predictor designed for instance set accuracy outputs the joint mode.\n" +
                "# The predictor designed for instance Hamming loss outputs the marginal modes.\n" +
                "# The predictor designed for instance F1 runs the GFM algorithm.\n" +
                "# The metric specified here will serve as the main metric for selecting the best model during hyper parameter tuning\n" +
                "# Once the model is trained, the program shows all different predictions made by different prediction methods\n" +
                "tune.targetMetric=instance_set_accuracy\n" +
                "\n" +
                "# the overall elastic-net penalty is a weighted combination of L1 norm and L2 norm and has the form\n" +
                "# penalty*[l1Ratio*L1 norm + (1-l1Ratio)*L2 norm]\n" +
                "\n" +
                "# What values to try for the overall elastic-net penalty\n" +
                "# Big values indicate strong regularizations\n" +
                "# The penalty can greatly affect the performance and thus requires careful tuning\n" +
                "tune.penalty.candidates=0.0001,0.000001\n" +
                "\n" +
                "# What values to try for L1 Ratio\n" +
                "# Any real number from 0 to 1, where 0 means L2 only, 1 means L1 only, and 0.5 means half L1 and half L2.\n" +
                "tune.l1Ratio.candidates=0.1,0.5\n" +
                "\n" +
                "# What values to try for number of CBM components\n" +
                "# The default value 50 usually gives good performance\n" +
                "# To reduce turning time, users can just set tune.numComponents.candidates=50\n" +
                "tune.numComponents.candidates=1\n" +
                "\n" +
                "\n" +
                "# Evaluate the metric on the validation set every k iterations\n" +
                "# Frequent evaluation may slow down training\n" +
                "# Use a small value (e.g. 1) if we expect the training to take just a few iterations (e.g. 20)\n" +
                "# Use a big value (e.g. 10) if we expect the training to take many iterations (e.g. 200)\n" +
                "tune.monitorInterval=1\n" +
                "\n" +
                "# the model training will never stop before it reaches this minimum number of iterations\n" +
                "tune.earlyStop.minIterations=5\n" +
                "\n" +
                "# If the validation metric does not improve after k successive evaluations, the training will stop\n" +
                "# for example, if tune.monitorInterval=5 and tune.earlyStop.patience=2, trains stops if no improvement in 10 iterations\n" +
                "# Using a patient value too small make cause the training to stop too early\n" +
                "# Using a patient value too big make increase the tuning time\n" +
                "tune.earlyStop.patience=10\n" +
                "\n" +
                "\n" +
                "######### train #################\n" +
                "\n" +
                "# Whether to use optimal hyper parameter values found by tuning\n" +
                "# These hyper parameters include: train.iterations, train.penalty, train.l1Ratio, and train.numComponents\n" +
                "# if true, users do not need to specify these values\n" +
                "# if false or if no tuning has be performed, users need to provide a value for each of them\n" +
                "train.useTunedHyperParameters=false\n" +
                "\n" +
                "# Number of EM training iterations\n" +
                "train.iterations=1\n" +
                "\n" +
                "# the overall elastic-net penalty is a weighted combination of L1 norm and L2 norm and has the form\n" +
                "# penalty*[l1Ratio*L1 norm + (1-l1Ratio)L2 norm]\n" +
                "# Big values indicate strong regularizations\n" +
                "# The penalty can greatly affect the performance and thus requires careful tuning\n" +
                "train.penalty=1E-4\n" +
                "\n" +
                "# Any real number from 0 to 1, where 0 means L2 only, 1 means L1 only, and 0.5 means half L1 and half L2.\n" +
                "train.l1Ratio=0\n" +
                "\n" +
                "# Number of CBM components\n" +
                "# The default value 50 usually gives good performance\n" +
                "train.numComponents=1\n" +
                "\n" +
                "\n" +
                "# The parameters below usually do not affect the performance much\n" +
                "# Users can use default values\n" +
                "\n" +
                "# whether to initialize CBM by random parameters\n" +
                "# default is false and uses BM to initialize CBM\n" +
                "train.randomInitialize=false\n" +
                "\n" +
                "# whether to use line search for elastic-net training\n" +
                "# using line search slows down training\n" +
                "# default=false\n" +
                "# in very rare occasions, the training may diverge without line search; if this happens, set lineSearch=true\n" +
                "train.elasticnet.lineSearch=false\n" +
                "\n" +
                "# whether to speed up elastic-net training using the active set trick\n" +
                "train.elasticnet.activeSet=true\n" +
                "\n" +
                "# number of coordinate descent iterations for LR in each M step\n" +
                "# The default value 5 is good most of the time\n" +
                "# If the train.iterations found by hyper parameter tuning is 1 or 2, each M step is probably doing too much work and the training overfits too quickly. In this case, we can decrease train.updatesPerIteration\n" +
                "train.updatesPerIteration=1\n" +
                "\n" +
                "# In each component, skip instances with small memberships values (gammas)\n" +
                "# This is designed to speed up training\n" +
                "train.skipDataThreshold=0.00001\n" +
                "\n" +
                "# Skip training a classifier for a label in a component if that label almost never appears or almost always appears in that component.\n" +
                "# A constant output (the prior probability) will be used in this case.\n" +
                "# This is designed to speed up training\n" +
                "train.skipLabelThreshold=0.00001\n" +
                "\n" +
                "# Smooth the probability of a non-existent label in a component with the its overall probability in the dataset\n" +
                "# This is designed to avoid zero probabilities\n" +
                "train.smoothStrength=0.0001\n" +
                "\n" +
                "######## test ##############\n" +
                "\n" +
                "# When generating prediction reports for individual label probabilities, labels with probabilities below the threshold will not be displayed\n" +
                "# This only make the reports more readable; it does not affect the actual prediction in any way.\n" +
                "report.labelProbThreshold=0.2\n" +
                "\n" +
                "\n" +
                "\n" +
                "\n" +
                "\n" +
                "# the internal Java class name for this application.\n" +
                "# users do not need to modify this.\n" +
                "pyramid.class=CBMEN";
        return Config.newConfigFromString(de);
    }

    private static class CalibInfo{
        private double uncalibrated;
        private double calibrated;
        private double accuracy;
    }
}
