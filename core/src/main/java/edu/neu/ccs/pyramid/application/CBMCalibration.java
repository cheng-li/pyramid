package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.AccPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.BucketInfo;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CBMCalibration {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.valid"), DataSetType.ML_CLF_SEQ_SPARSE,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.test"), DataSetType.ML_CLF_SEQ_SPARSE,true);
        CBM cbm = (CBM) Serialization.deserialize(config.getString("input.cbm"));
        List<MultiLabel> support = (List) Serialization.deserialize(config.getString("input.support"));
        System.out.println("calibrated probability");
        calibrated(cbm, support, valid, test);
        System.out.println("uncalibrated probability");
        uncalibrated(cbm,test);

    }


    private static void calibrated(CBM cbm, List<MultiLabel> support, MultiLabelClfDataSet valid, MultiLabelClfDataSet test){
        IsotonicRegression isotonicRegression = trainIso(cbm, support, valid);
        AccPredictor accPredictor = new AccPredictor(cbm);
        Stream<Pair<Double,Integer>> stream = IntStream.range(0, test.getNumDataPoints()).parallel().mapToObj(i->{
            MultiLabel pre = accPredictor.predict(test.getRow(i));
            double pro = cbm.predictAssignmentProb(test.getRow(i),pre,0.001);
            int correct = 0;
            if (pre.equals(test.getMultiLabels()[i])){
                correct = 1;
            }
            Pair<Double,Integer>  pair = new Pair<>(pro,correct);
            return pair;
        });
        System.out.println(isotonicRegression.displayCalibrationResult(stream));
    }

    private static void uncalibrated(CBM cbm, MultiLabelClfDataSet test){
        AccPredictor accPredictor = new AccPredictor(cbm);
        Stream<Pair<Double,Integer>> stream = IntStream.range(0, test.getNumDataPoints()).parallel().mapToObj(i->{
            MultiLabel pre = accPredictor.predict(test.getRow(i));
            double pro = cbm.predictAssignmentProb(test.getRow(i),pre,0.001);
            int correct = 0;
            if (pre.equals(test.getMultiLabels()[i])){
                correct = 1;
            }
            Pair<Double,Integer>  pair = new Pair<>(pro,correct);
            return pair;
        });

        System.out.println(display(stream));
    }

    private static IsotonicRegression trainIso(CBM cbm, List<MultiLabel> support, MultiLabelClfDataSet valid) {
        AccPredictor accPredictor = new AccPredictor(cbm);
        Stream<Pair<Double, Integer>> stream = IntStream.range(0, valid.getNumDataPoints()).parallel().boxed().
                flatMap(i -> {
                    MultiLabel pre = accPredictor.predict(valid.getRow(i));
                    Set<MultiLabel> copy = new HashSet<>(support);
                    if (!copy.contains(pre)) {
                        copy.add(pre);
                    }
                    List<MultiLabel> candidate = new ArrayList<>(copy);
                    double[] probs = cbm.predictAssignmentProbs(valid.getRow(i), candidate, 0.001);
                    Stream<Pair<Double, Integer>> pairs = IntStream.range(0, candidate.size()).mapToObj(c -> {
                        double pro = probs[c];
                        int correct = 0;
                        if (candidate.get(c).equals(valid.getMultiLabels()[i])) {
                            correct = 1;
                        }
                        return new Pair<Double, Integer>(pro, correct);
                    });
                    return pairs;
                });
        IsotonicRegression isotonicRegression = new IsotonicRegression(stream);
        return isotonicRegression;
    }

    public static String display(Stream<Pair<Double, Integer>> stream){
        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = stream.map(doubleIntegerPair -> {
            double probs = doubleIntegerPair.getFirst();
            double[] sum = new double[numBuckets];
            double[] sumProbs = new double[numBuckets];
            double[] count = new double[numBuckets];
            int index = (int)Math.floor(probs/bucketLength);
            if (index<0){
                index=0;
            }
            if (index>=numBuckets){
                index = numBuckets-1;
            }
            count[index] += 1;
            sumProbs[index] += probs;
            sum[index]+=doubleIntegerPair.getSecond();
            return new BucketInfo(count, sum,sumProbs);
        }).reduce(empty, BucketInfo::add, BucketInfo::add);
        double[] counts = total.getCounts();
        double[] correct = total.getSums();
        double[] sumProbs = total.getSumProbs();
        double[] accs = new double[counts.length];
        double[] average_confidence = new double[counts.length];

        for (int i = 0; i < counts.length; i++) {
            accs[i] = correct[i] / counts[i];
        }
        for (int j = 0; j < counts.length; j++) {
            average_confidence[j] = sumProbs[j] / counts[j];
        }

        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
        for (int i = 0; i < 10; i++) {
            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
                    .append(counts[i] - correct[i]).append("\t\t").append(decimalFormat.format(accs[i])).append("\t\t")
                    .append(decimalFormat.format(average_confidence[i])).append("\n");

        }

        String result = sb.toString();
        return result;

    }
}
