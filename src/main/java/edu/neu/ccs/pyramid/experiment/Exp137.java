package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.simulation.MultiLabelSynthesizer;
import edu.neu.ccs.pyramid.util.Grid;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.neu.ccs.pyramid.util.Translator;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 11/3/15.
 */
public class Exp137 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        dump();
//        lr_lasso();

//        lr();
//        boost();
//         checkIdeal();
        mix_da();
//        check_mix_da();
    }


    private static void dump() throws Exception{
        MultiLabelClfDataSet all = MultiLabelSynthesizer.flipOneNonUniform(10000);
        Pair<ClfDataSet,Translator<MultiLabel>> pair = DataSetUtil.toMultiClass(all);
        ClfDataSet clf = pair.getFirst();
        Translator<MultiLabel> translator = pair.getSecond();
        List<Integer> trainIndices = IntStream.range(0, 5000).mapToObj(i -> i).collect(Collectors.toList());
        List<Integer> testIndices = IntStream.range(5000,10000).mapToObj(i->i).collect(Collectors.toList());
        ClfDataSet trainSet = DataSetUtil.sampleData(clf, trainIndices);
        ClfDataSet testSet =  DataSetUtil.sampleData(clf, testIndices);
        TRECFormat.save(trainSet, new File(TMP, "train.trec"));
        TRECFormat.save(testSet, new File(TMP,"test.trec"));
        LogisticRegression logisticRegression = ideal(translator);
        System.out.println(logisticRegression);
        Serialization.serialize(logisticRegression,new File(TMP,"model"));
        System.out.println("train acc="+Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println("test acc="+Accuracy.accuracy(logisticRegression,testSet));
    }

    private static void lr() throws Exception{
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(new File(TMP,"train.trec"), DataSetType.CLF_DENSE,true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(TMP,"test.trec"), DataSetType.CLF_DENSE,true);

        LogisticRegression logisticRegression = new LogisticRegression(trainSet.getNumClasses(),trainSet.getNumFeatures());
        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(logisticRegression,trainSet,100000);
//        optimizer.getOptimizer().getTerminator().setMaxIteration(100).setMode(Terminator.Mode.FINISH_MAX_ITER);
        optimizer.setParallelism(true);
        optimizer.optimize();

        System.out.println("train acc = "+ Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println("test acc = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("train log likelihood = "+logisticRegression.dataSetLogLikelihood(trainSet));
        System.out.println(logisticRegression);

        int[] predictions = logisticRegression.predict(testSet);
        StringBuilder sb = new StringBuilder();
        for (int p: predictions){
            sb.append(p).append("\n");
        }

        FileUtils.writeStringToFile(new File(TMP,"lr_pred"),sb.toString());

    }

    private static void lr_lasso() throws Exception{
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(new File(TMP,"train.trec"), DataSetType.CLF_DENSE,true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(TMP,"test.trec"), DataSetType.CLF_DENSE,true);

        LogisticRegression logisticRegression = new LogisticRegression(trainSet.getNumClasses(),trainSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = new ElasticNetLogisticTrainer.Builder(logisticRegression,trainSet)
                .setL1Ratio(1)
                .setRegularization(0.00001).build();
        trainer.optimize();

        System.out.println("train acc = "+ Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println("test acc = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("train log likelihood = "+logisticRegression.dataSetLogLikelihood(trainSet));
        System.out.println(logisticRegression);
    }

    private static void boost() throws Exception{
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(new File(TMP,"train.trec"), DataSetType.CLF_DENSE,true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(TMP,"test.trec"), DataSetType.CLF_DENSE,true);

        LKBoost lkBoost = new LKBoost(trainSet.getNumClasses());


        RegTreeConfig treeConfig = new RegTreeConfig();
        treeConfig.setMaxNumLeaves(10);
        RegTreeFactory factory = new RegTreeFactory(treeConfig);
        factory.setLeafOutputCalculator(new LKBOutputCalculator(trainSet.getNumClasses()));
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,trainSet,factory);

        trainer.setShrinkage(0.1);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round =0;round<20;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println("train acc = "+ Accuracy.accuracy(lkBoost,trainSet));
            System.out.println("test acc = "+ Accuracy.accuracy(lkBoost,testSet));
        }

        int[] predictions = lkBoost.predict(testSet);
        StringBuilder sb = new StringBuilder();
        for (int p: predictions){
            sb.append(p).append("\n");
        }

        FileUtils.writeStringToFile(new File(TMP,"boost_pred"),sb.toString());

    }

    private static LogisticRegression ideal(Translator<MultiLabel> translator){
        LogisticRegression logisticRegression = new LogisticRegression(16,2);
        for (int l=0;l<16;l++){
            MultiLabel multiLabel = translator.getObj(l);
            Vector vector = map(multiLabel);
            logisticRegression.getWeights().getWeightsWithoutBiasForClass(l).set(0,vector.get(0));
            logisticRegression.getWeights().getWeightsWithoutBiasForClass(l).set(1,vector.get(1));
        }
        return logisticRegression;


    }

    private static Vector map(MultiLabel multiLabel){
        Vector vector0 = new DenseVector(2);
        vector0.set(0,0);
        vector0.set(1,1);

        Vector vector1 = new DenseVector(2);
        vector1.set(0,1);
        vector1.set(1,1);

        Vector vector2 = new DenseVector(2);
        vector2.set(0,1);
        vector2.set(1,0);

        Vector vector3 = new DenseVector(2);
        vector3.set(0,1);
        vector3.set(1,-1);

        List<Vector> vectors = new ArrayList<>();
        vectors.add(vector0);
        vectors.add(vector1);
        vectors.add(vector2);
        vectors.add(vector3);

        Vector vector = new DenseVector(2);

        if (multiLabel.matchClass(0)){
            vector = vector.minus(vectors.get(0));
        } else {
            vector = vector.plus(vectors.get(0));
        }


        for (int l=1;l<4;l++){
            if (multiLabel.matchClass(l)){
                vector = vector.plus(vectors.get(l));
            } else {
                vector = vector.minus(vectors.get(l));
            }
        }
        return vector;
    }

    public static void checkIdeal() throws Exception{
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(new File(TMP,"train.trec"), DataSetType.CLF_DENSE,true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(TMP,"test.trec"), DataSetType.CLF_DENSE,true);
        LogisticRegression logisticRegression = (LogisticRegression)Serialization.deserialize(new File(TMP,"model"));
        System.out.println("initialize with ideal logistic regression");

        System.out.println("train acc = "+ Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println("test acc = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("train log likelihood = "+logisticRegression.dataSetLogLikelihood(trainSet));


        int[] predictions = logisticRegression.predict(testSet);
        StringBuilder sb = new StringBuilder();
        for (int p: predictions){
            sb.append(p).append("\n");
        }

        FileUtils.writeStringToFile(new File(TMP,"ideal_pred"),sb.toString());


        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(logisticRegression,trainSet,100);
        optimizer.setParallelism(true);
        optimizer.optimize();

        System.out.println("after optimization");
        System.out.println("train acc = "+ Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println("test acc = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("train log likelihood = "+logisticRegression.dataSetLogLikelihood(trainSet));

    }

    private static void mix_da() throws Exception{
        System.out.println("mix deterministic annealing");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);
        int numClusters = 2;
        double softmaxVariance = 100;
        double logitVariance = 100;

        List<Double> grid = Grid.uniform(0.7,1,10);

        BMMClassifier bmmClassifier = BMMClassifier.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setBinaryClassifierType("lr")
                .setMultiClassClassifierType("lr")
                .build();


        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);

        for (double it: grid){
            BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
            optimizer.setInverseTemperature(it);
            System.out.println("inverse temperature = "+it);
            for (int i=1;i<=10;i++){
                optimizer.iterate();
                MultiLabel[] trainPredict;
                MultiLabel[] testPredict;
                trainPredict = bmmClassifier.predict(trainSet);
                testPredict = bmmClassifier.predict(testSet);
                System.out.print("iter : "+i + "\t");
                System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
                System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
                System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
                System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
                System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
                System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
                System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
                if (optimizer.getTerminator().shouldTerminate()){
                    break;
                }
            }
            double[][] gammas = optimizer.getGammas();
            double perplexity=    IntStream.range(0,gammas.length).mapToDouble(i->Math.exp(Entropy.entropy(gammas[i]))).average().getAsDouble();
            System.out.println("perplexity of gammas = "+perplexity);


            List<double[]> list = bmmClassifier.getMultiClassClassifier().predictClassProbs(trainSet);
            double perplexitypi=    IntStream.range(0,list.size()).mapToDouble(i->Math.exp(Entropy.entropy(list.get(i)))).average().getAsDouble();
            System.out.println("perplexity of pi= "+perplexitypi);

            for (int k=0;k<numClusters;k++){
                System.out.println("cluster = "+k);
                for (int i=0;i<trainSet.getNumDataPoints();i++){
                    if (gammas[i][k]>0.7){
                        System.out.println(trainSet.getMultiLabels()[i]);
                    }
                }
            }


        }

        Serialization.serialize(bmmClassifier,new File(TMP,"mix"));


//        System.out.println(bmmClassifier);
    }

    private static void check_mix_da() throws Exception {
        System.out.println("mix deterministic annealing");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);

        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize(new File(TMP, "mix"));
        List<double[]> list = bmmClassifier.getMultiClassClassifier().predictClassProbs(trainSet);
        double perplexity=    IntStream.range(0,list.size()).mapToDouble(i->Math.exp(Entropy.entropy(list.get(i)))).average().getAsDouble();
        System.out.println("perplexity of pi= "+perplexity);



    }


}
