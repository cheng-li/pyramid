package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.RMSE;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/21/17.
 */
public class EMLevelEval {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        RegDataSet train = TRECFormat.loadRegDataSet(config.getString("input.trainData"), DataSetType.REG_SPARSE,true);

        Set<Double> unique = new HashSet<>();
        for (double d: train.getLabels()){
            unique.add(d);
        }

        List<Double> levels = unique.stream().sorted().collect(Collectors.toList());

        RegDataSet test = TRECFormat.loadRegDataSet(config.getString("input.testData"), DataSetType.REG_SPARSE,true);

        double[] doubleTruth = test.getLabels();

        double[] doublePred = loadPrediction(config.getString("input.prediction"));
        double[] roundedPred = Arrays.stream(doublePred).map(d->round(d,levels)).toArray();

        System.out.println("before rounding");
        System.out.println("rmse = "+ RMSE.rmse(doubleTruth, doublePred));

        System.out.println("after rounding");
        System.out.println("rmse = "+ RMSE.rmse(doubleTruth, roundedPred));
        System.out.println("accuracy = "+ IntStream.range(0, test.getNumDataPoints()).filter(i->doubleTruth[i]==roundedPred[i]).count()/(double)test.getNumDataPoints());


        System.out.println("the distribution of predicted label for a given true label");
        for (int l=0;l<levels.size();l++){
            double level = levels.get(l);
            System.out.println("for true label "+level);

            truthToPred(test.getLabels(),roundedPred, level, levels);
        }

        System.out.println("=============================");

        System.out.println("the distribution of true label for a given predicted label");
        for (int l=0;l<levels.size();l++){
            double level = levels.get(l);
            System.out.println("for predicted label "+level);

            predToTruth(test.getLabels(),roundedPred, level, levels);
        }


    }

    private static double[] loadPrediction(String file) throws IOException {
        return FileUtils.readLines(new File(file)).stream().mapToDouble(a->Double.parseDouble(a)).toArray();
    }

    private static double round(double d, List<Double> levels){
        double mindis = Double.POSITIVE_INFINITY;
        double res = 0;
        for (double level: levels){
            double dis = Math.abs(level-d);
            if (dis<mindis){
                res = level;
                mindis = dis;
            }
        }
        return res;
    }

    private static double[] count(double[] input, List<Double> levels){
        double[] count = new double[levels.size()];

        for (int l=0;l<levels.size();l++){
            double level = levels.get(l);
            count[l] = Arrays.stream(input).filter(d->d==level).count();
        }

        for (int i=0;i<count.length;i++){
            count[i] /= input.length;
        }
        return count;
    }


    private static void truthToPred(double[] truth, double[] pred, double target, List<Double> levels){
        double[] filtered = IntStream.range(0, truth.length).filter(i-> truth[i]==target).mapToDouble(i->pred[i]).toArray();
        double[] count = count(filtered, levels);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<levels.size();l++){
            double level = levels.get(l);
            sb.append(level).append(":").append(count[l]).append(", ");
        }
        System.out.println(sb.toString());
    }


    private static void predToTruth(double[] truth, double[] pred, double target, List<Double> levels){
        double[] filtered = IntStream.range(0, truth.length).filter(i-> pred[i]==target).mapToDouble(i->truth[i]).toArray();
        double[] count = count(filtered, levels);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<levels.size();l++){
            double level = levels.get(l);
            sb.append(level).append(":").append(count[l]).append(", ");
        }
        System.out.println(sb.toString());
    }


}
