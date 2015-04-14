package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Pair;
import edu.stanford.nlp.parser.nndep.Dataset;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * step-wise logistic regression on a given matrix
 * Created by chengli on 4/12/15.
 */
public class Exp81 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        train(config);


    }


    private static void train(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet fullTrainData = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);
        System.out.println("full training data loaded");

        ClfDataSet testData = TRECFormat.loadClfDataSet(new File(input, config.getString("input.testData")),
                DataSetType.CLF_SPARSE, true);
        System.out.println("test data loaded");

        ClfDataSet trainData = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);

        for (int i=0;i<trainData.getNumDataPoints();i++){
            Vector row = fullTrainData.getRow(i);
            for (Vector.Element element: row.nonZeroes()){
                int featureIndex = element.index();
                trainData.setFeatureValue(i,featureIndex,0);
            }
        }

        System.out.println("empty training set created");

        LogisticRegression logisticRegression = new LogisticRegression(trainData.getNumClasses(),trainData.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,trainData)
                .setEpsilon(0.01).setL1Ratio(config.getDouble("l1Ratio"))
                .setRegularization(config.getDouble("regularization")).build();


        Set<Integer> remainingFeatures = new HashSet<>();
        for (int i=0;i<fullTrainData.getNumFeatures();i++){
            remainingFeatures.add(i);
        }

        Set<Integer> usedFeatures = new HashSet<>();


        int iteration = 0;
        while(true){
            System.out.println("iteration = "+iteration);
            trainer.train();
            System.out.println("number of features = "+usedFeatures.size());
            System.out.println("accuracy on training set = "+ Accuracy.accuracy(logisticRegression,trainData));
            System.out.println("accuracy on test set = "+ Accuracy.accuracy(logisticRegression,testData));

            if (remainingFeatures.size()==0){
                break;
            }


            int numFeaturesToAdd = config.getInt("numFeaturesToAdd");
            List<double[]> probs = logisticRegression.predictClassProbs(trainData);

            Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);

            IntStream.range(0,trainData.getNumClasses()).forEach(k->{
                List<Integer> toAdd = remainingFeatures.stream().parallel().map(i -> {
                    double utility = utility(fullTrainData, k, i, probs);
                    return new Pair<Integer, Double>(i, utility);
                }).sorted(comparator.reversed()).
                        map(Pair::getFirst).
                        limit(numFeaturesToAdd).collect(Collectors.toList());

                toAdd.stream().parallel().forEach(featureIndex -> copyColumn(fullTrainData,trainData,featureIndex));

                usedFeatures.addAll(toAdd);
                remainingFeatures.removeAll(toAdd);
            });



            iteration += 1;
        }


    }


    private static void copyColumn(DataSet source, DataSet des, int featureIndex){
        Vector column = source.getColumn(featureIndex);
        for (Vector.Element element: column.nonZeroes()){
            int dataIndex = element.index();
            double value = element.get();
            des.setFeatureValue(dataIndex,featureIndex,value);
        }
    }

    private static double utility(ClfDataSet dataSet, int classIndex, int featureIndex, List<double[]> probs){
        Vector vector = dataSet.getColumn(featureIndex);
        int[] labels = dataSet.getLabels();
        //actual and predicted
        double[] counts = new double[2];
        for (Vector.Element element: vector.nonZeroes()){
            int dataPoint = element.index();
            counts[1] += probs.get(dataPoint)[classIndex];
            if (labels[dataPoint]==classIndex){
                counts[0] += 1;
            }
        }
        return counts[0] - counts[1];
    }
}
