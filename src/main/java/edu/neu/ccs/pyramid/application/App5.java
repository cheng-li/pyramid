package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBMOptimizer;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.PluginF1;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


/**
 * CBM
 * Created by Rainicy on 10/24/15.
 */
public class App5 {

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
        String matrixType = config.getString("input.matrixType");

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;

        switch (matrixType){
            case "sparse_random":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SPARSE, true);
                break;
            case "sparse_sequential":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                break;
            case "dense":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_DENSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_DENSE, true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }


        int numIterations = config.getInt("em.numIterations");

        String output = config.getString("output");
        String modelName = "model";
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();


        Pair<CBM,Integer> pair = loadCBM(config,trainSet);
        CBM cbm = pair.getFirst();


        CBMOptimizer optimizer = getOptimizer(config, cbm, trainSet);

        PluginF1 pluginF1 = new PluginF1(cbm);
        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
        pluginF1.setSupport(support);
        MultiLabelClassifier classifier;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                classifier = cbm;
                break;
            case "instanceFMeasure":
                classifier = pluginF1;
                break;
            default:
                throw new IllegalArgumentException("predictTarget can be subsetAccuracy or instanceFMeasure");
        }

        for (int i=1;i<=numIterations;i++){
            System.out.println("=================================================");
            System.out.println("iteration : "+i );
            optimizer.iterate();
            System.out.println("loss: "+optimizer.getTerminator().getLastValue());

            System.out.println("training performance with "+predictTarget+" optimal predictor:");
            System.out.println(new MLMeasures(classifier,trainSet));
            System.out.println("test performance with "+predictTarget+" optimal predictor:");
            System.out.println(new MLMeasures(classifier,testSet));

            File serializeModel = new File(path,  "iter." + i + ".model");
            cbm.serialize(serializeModel);

        }
        File serializeModel = new File(path, "model");
        cbm.serialize(serializeModel);
    }

    private static void test(Config config) throws Exception{


        String matrixType = config.getString("input.matrixType");

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;

        switch (matrixType){
            case "sparse_random":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SPARSE, true);
                break;
            case "sparse_sequential":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                break;
            case "dense":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_DENSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_DENSE, true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }
        String output = config.getString("output");
        String modelName = "model";
        File path = Paths.get(output, modelName).toFile();
        CBM cbm = (CBM) Serialization.deserialize(new File(path, "model"));
        PluginF1 pluginF1 = new PluginF1(cbm);
        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
        pluginF1.setSupport(support);

        String predictTarget = config.getString("predict.target");
        MultiLabelClassifier classifier;
        switch (predictTarget){
            case "subsetAccuracy":
                classifier = cbm;
                break;
            case "instanceFMeasure":
                classifier = pluginF1;
                break;
            default:
                throw new IllegalArgumentException("predictTarget can be subsetAccuracy or instanceFMeasure");
        }
        System.out.println("test performance with "+predictTarget+" optimal predictor:");
        System.out.println(new MLMeasures(classifier,testSet));

    }

    private static CBMOptimizer getOptimizer(Config config, CBM cbm, MultiLabelClfDataSet trainSet){
        CBMOptimizer optimizer = new CBMOptimizer(cbm,trainSet);

//        optimizer.setLineSearch(config.getBoolean("elasticnet.lineSearch"));
//        optimizer.setRegularizationBinary(config.getDouble("elasticnet.binaryRegularization"));
//        optimizer.setRegularizationMultiClass(config.getDouble("elasticnet.multiClassRegularization"));
//        optimizer.setL1RatioBinary(config.getDouble("elasticnet.binaryL1Ratio"));
//        optimizer.setL1RatioMultiClass(config.getDouble("elasticnet.multiClassL1Ratio"));

        optimizer.setPriorVarianceMultiClass(config.getDouble("lr.multiClassVariance"));
        optimizer.setPriorVarianceBinary(config.getDouble("lr.binaryVariance"));
        optimizer.setNumIterationsBinary(config.getInt("boost.numIterationsBinary"));
        optimizer.setNumIterationsMultiClass(config.getInt("boost.numIterationsMultiClass"));
        optimizer.setShrinkageBinary(config.getDouble("boost.shrinkageBinary"));
        optimizer.setShrinkageMultiClass(config.getDouble("boost.shrinkageMultiClass"));
        optimizer.setNumLeavesBinary(config.getInt("boost.numLeavesBinary"));
        optimizer.setNumLeavesMultiClass(config.getInt("boost.numLeavesMultiClass"));

        return optimizer;
    }

    private static Pair<CBM,Integer> loadOldCBM(Config config) throws Exception{

        CBM cbm;
        int completedIterations = 0;
        String output = config.getString("output");
        String modelName = "model";
        File folder = Paths.get(output,modelName).toFile();
        File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")));
        File lastFile = null;
        int lastIter = -1;
        for (File file: modeFiles){
            String[] split = file.getName().split(Pattern.quote("."));
            int iter = Integer.parseInt(split[1]);
            if (iter>lastIter){
                lastIter = iter;
                lastFile = file;
                completedIterations = lastIter;
            }
        }
        cbm = (CBM) Serialization.deserialize(lastFile);
        System.out.println("cbm loaded, with "+completedIterations+ " iterations completed");
        return new Pair<>(cbm,completedIterations);
    }

    private static Pair<CBM,Integer> loadNewCBM(Config config, MultiLabelClfDataSet trainSet) throws Exception{

        CBM cbm;
        int completedIterations = 0;

        cbm = CBM.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumComponents(config.getInt("mixture.numComponents"))
                .setMultiClassClassifierType(config.getString("mixture.multiClassClassifierType"))
                .setBinaryClassifierType(config.getString("mixture.binaryClassifierType"))
                .build();


        String allowEmpty = config.getString("predict.allowEmpty");
        switch (allowEmpty){
            case "true":
                cbm.setAllowEmpty(true);
                break;
            case "false":
                cbm.setAllowEmpty(false);
                break;
            case "auto":
                Set<MultiLabel> seen = DataSetUtil.gatherMultiLabels(trainSet).stream().collect(Collectors.toSet());
                MultiLabel empty = new MultiLabel();
                if (seen.contains(empty)){
                    cbm.setAllowEmpty(true);
                    System.out.println("training set contains empty labels, automatically set predict.allowEmpty = true");
                } else {
                    cbm.setAllowEmpty(false);
                    System.out.println("training set does not contain empty labels, automatically set predict.allowEmpty = false");
                }
                break;
            default:
                throw new IllegalArgumentException("unknown value for predict.allowEmpty");
        }

        if (config.getBoolean("train.initialize")) {
            System.out.println("start initialization ");
            CBMOptimizer optimizer = getOptimizer(config, cbm,trainSet);
            CBMInitializer.initialize(cbm, trainSet, optimizer);
            System.out.println("finish initialization");
        } else {
            System.out.println("no initialization is used, use random initial weights");
            CBMOptimizer optimizer = getOptimizer(config, cbm,trainSet);
            CBMInitializer.randInitialize(cbm, trainSet, optimizer);
        }

        System.out.println("cbm loaded, with "+completedIterations+ " iterations completed");
        return new Pair<>(cbm,completedIterations);
    }



    private static Pair<CBM,Integer> loadCBM(Config config, MultiLabelClfDataSet trainSet) throws Exception{
        String mode = config.getString("train.warmStart");
        Pair<CBM,Integer> pair = null;
        switch (mode){
            case "true":
                pair = loadOldCBM(config);
                break;
            case "false":
                pair = loadNewCBM(config,trainSet);
                break;
            case "auto":
                String output = config.getString("output");
                String modelName = "model";
                File folder = Paths.get(output,modelName).toFile();
                File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")));
                if (modeFiles.length==0){
                    pair = loadNewCBM(config,trainSet);
                } else {
                    pair = loadOldCBM(config);
                }
                break;
            default:
                throw new IllegalArgumentException("unknown value for train.warmStart");
        }
        return pair;
    }


}