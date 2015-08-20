package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by Rainicy on 8/20/15.
 */
public class Exp206 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new RuntimeException("please give a config file");
        }

        Config config = new Config(args[0]);

        String inputFolder = config.getString("input.folder");
        String trainData = config.getString("train");
        String testData = config.getString("test");

        File trainFile = new File(new File(inputFolder, "data_sets"),trainData);
        File testFile = new File(new File(inputFolder, "data_sets"),testData);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(trainFile, DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(testFile, DataSetType.ML_CLF_SPARSE, true);

        MultiLabel[] trainLabels = trainSet.getMultiLabels();
        MultiLabel[] testLabels = testSet.getMultiLabels();

        MultiLabel[] wholeLabels = new MultiLabel[trainLabels.length + testLabels.length];
        for (int i=0; i<wholeLabels.length; i++) {
            if (i < trainLabels.length) {
                wholeLabels[i] = trainLabels[i];
            } else {
                wholeLabels[i] = testLabels[i-trainLabels.length];
            }
        }
        System.out.println("Done with loading dataset.");

        // sampling an random index list
        List<Integer> indexList = new LinkedList<>();
        for (int i=0; i<wholeLabels.length; i++) {
            indexList.add(i);
        }
        Collections.shuffle(indexList);
        System.out.println("Done with sampling list.");


        int splitNum = config.getInt("split.number");
        String output = config.getString("output.File");

        BufferedWriter bw = new BufferedWriter(new FileWriter(output));

        System.out.println("Starting splitting...");
        Set<Set<Integer>> multiLabelSet = new HashSet<>();
        int step = wholeLabels.length / splitNum;
        for (int i=0; i<splitNum; i++) {
            System.out.println(i + "/" + splitNum);
            int startIndex = step * i;
            int stopIndex = step * (i+1);

            for (int j=startIndex; j<stopIndex; j++) {
                MultiLabel label = wholeLabels[j];
                Set<Integer> setLabel = label.getMatchedLabels();
                multiLabelSet.add(setLabel);
            }

            bw.write(stopIndex + "\t" + multiLabelSet.size() + "\n");
        }
        bw.close();

    }
}
