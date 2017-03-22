package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.RMSE;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
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

        ClfDataSet test = TRECFormat.loadClfDataSet(config.getString("input.testData"), DataSetType.CLF_SPARSE,true);

        double[] doubleTruth = Arrays.stream(test.getLabels()).mapToDouble(a->a).toArray();

        double[] doublePred = loadPrediction(config.getString("input.prediction"));
        int[] roundedPred = Arrays.stream(doublePred).mapToInt(d->round(d,0,test.getNumClasses()-1)).toArray();
        double[] roundPredAsDouble = Arrays.stream(roundedPred).mapToDouble(a->a).toArray();

        System.out.println("before rounding");
        System.out.println("rmse = "+ RMSE.rmse(doubleTruth, doublePred));

        System.out.println("after rounding");
        System.out.println("rmse = "+ RMSE.rmse(doubleTruth, roundPredAsDouble));
        System.out.println("accuracy = "+ Accuracy.accuracy(test.getLabels(), roundedPred));


        System.out.println("the distribution of predicted label for a given true label");
        for (int l=0;l<test.getNumClasses();l++){
            System.out.println("for true label "+l);
            truthToPred(test.getLabels(),roundedPred, l, test.getNumClasses(), test.getLabelTranslator());
        }

        System.out.println("=============================");

        System.out.println("the distribution of true label for a given predicted label");
        for (int l=0;l<test.getNumClasses();l++){
            System.out.println("for predicted label "+l);
            predToTruth(test.getLabels(),roundedPred, l, test.getNumClasses(), test.getLabelTranslator());
        }


    }

    private static double[] loadPrediction(String file) throws IOException {
        return FileUtils.readLines(new File(file)).stream().mapToDouble(a->Double.parseDouble(a)).toArray();
    }

    private static int round(double d, int min, int max){
        int r = (int)Math.round(d);
        if (r<min){
            r=min;
        }
        if (r>max){
            r=max;
        }
        return r;
    }

    private static double[] count(int[] input, int numClasses){
        double[] count = new double[numClasses];
        for (int d: input){
            count[d] += 1;
        }

        for (int i=0;i<count.length;i++){
            count[i] /= input.length;
        }
        return count;
    }


    private static void truthToPred(int[] truth, int[] pred, int target, int numClasses, LabelTranslator labelTranslator){
        int[] filtered = IntStream.range(0, truth.length).filter(i-> truth[i]==target).map(i->pred[i]).toArray();
        double[] count = count(filtered, numClasses);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<numClasses;l++){
            sb.append(labelTranslator.toExtLabel(l)).append(":").append(count[l]).append(", ");
        }
        System.out.println(sb.toString());
    }


    private static void predToTruth(int[] truth, int[] pred, int target, int numClasses, LabelTranslator labelTranslator){
        int[] filtered = IntStream.range(0, truth.length).filter(i-> pred[i]==target).map(i->truth[i]).toArray();
        double[] count = count(filtered, numClasses);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<numClasses;l++){
            sb.append(labelTranslator.toExtLabel(l)).append(":").append(count[l]).append(", ");
        }
        System.out.println(sb.toString());
    }


}
