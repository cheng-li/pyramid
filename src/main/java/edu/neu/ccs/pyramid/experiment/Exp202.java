package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.reuters.IndexBuilder;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Rainicy on 6/27/15.
 */
public class Exp202 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);

        String path = config.getString("path");
        String codesDictPath = config.getString("codes.dict");
        Map<String, String> codesDictMap = IndexBuilder.getCodesDict(codesDictPath);


        List<String> labels = new ArrayList<>();
        for(Map.Entry<String, String> entry : codesDictMap.entrySet()) {
            String code = entry.getKey();
            String desc = entry.getValue();
            labels.add(code + " : " + desc);
        }

        // deserialize
        IMLGradientBoosting imlGradientBoosting = (IMLGradientBoosting) Serialization.deserialize(new File(path,"model"));
        // load data
        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "train"), DataSetType.ML_CLF_SPARSE ,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "test"), DataSetType.ML_CLF_SPARSE ,true);

        MultiLabel[] testTrue = test.getMultiLabels();
        MultiLabel[] trainTrue = train.getMultiLabels();
        List<MultiLabel> testPredict = imlGradientBoosting.predict(test);
        List<MultiLabel> trainPredict = imlGradientBoosting.predict(train);

        LabelTranslator labelTranslator = test.getLabelTranslator();


        Map<String, Double> trainAcc = new HashMap<>();
        Map<String, Double> testAcc = new HashMap<>();

//        int labelId = 0;
        for (String careLabel : labels) {
//            System.out.println(++labelId + ": " + careLabel);
            int careLabelIndex = 0;
            try {
                careLabelIndex = labelTranslator.toIntLabel(careLabel);
            } catch (Exception e) {
//                System.out.println("Missing label: " + careLabel);
//                labelId--;
                continue;
            }


            int count = 0;
            // get train acc
            for (int i=0; i<trainTrue.length; i++) {
                MultiLabel yHat = trainPredict.get(i);
                MultiLabel y = trainTrue[i];

                // predict correctly
                if ( (yHat.matchClass(careLabelIndex) && y.matchClass(careLabelIndex)) ||
                        (!yHat.matchClass(careLabelIndex) && !y.matchClass(careLabelIndex)) ) {
                    count++;
                }
            }
            double acc = (double) count / trainTrue.length;
            trainAcc.put(careLabel, acc);

            count = 0;
            // get test acc
            for (int i=0; i<testTrue.length; i++) {
                MultiLabel yHat = testPredict.get(i);
                MultiLabel y = testTrue[i];

                // predict correctly
                if ( (yHat.matchClass(careLabelIndex) && y.matchClass(careLabelIndex)) ||
                        (!yHat.matchClass(careLabelIndex) && !y.matchClass(careLabelIndex)) ) {
                    count++;
                } else if (careLabel.startsWith("C41 :")) {
                    System.out.println(i + ": " + test.getIdTranslator().toExtId(i));
                }
            }
            acc = (double) count / testTrue.length;
            testAcc.put(careLabel, acc);
        }

        String output = config.getString("output");
        String outputTrain = output + ".train.txt";
        String outputTest = output + ".test.txt";
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputTrain));
        bw.write("accuracy on training set = " + Accuracy.accuracy(trainTrue, trainPredict) + "\n");
        bw.write("overlap on training set = " + Overlap.overlap(trainTrue, trainPredict) + "\n");
        for (Map.Entry<String, Double> entry : trainAcc.entrySet()) {
            bw.write(entry.getKey() + "\t" + entry.getValue() + "\n");
        }
        bw.close();

        bw = new BufferedWriter(new FileWriter(outputTest));
        bw.write("accuracy on testing set = " + Accuracy.accuracy(testTrue, testPredict) + "\n");
        bw.write("overlap on testing set = " + Overlap.overlap(testTrue, testPredict) + "\n");
        for (Map.Entry<String, Double> entry : testAcc.entrySet()) {
            bw.write(entry.getKey() + "\t" + entry.getValue() + "\n");
        }
        bw.close();

    }
}
