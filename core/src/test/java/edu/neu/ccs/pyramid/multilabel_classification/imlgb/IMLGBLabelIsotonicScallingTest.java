package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.text.DecimalFormat;

public class IMLGBLabelIsotonicScallingTest {
    public static void main(String[] args) throws Exception {
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        MultiLabelClfDataSet multiLabelClfDataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("test"), DataSetType.ML_CLF_SPARSE, true);
        IMLGradientBoosting imlGradientBoosting = (IMLGradientBoosting) Serialization.deserialize(config.getString("model_app3"));
        IMLGBJointLabelIsotonicScaling imlgbJointLabelIsotonicScaling = new IMLGBJointLabelIsotonicScaling(imlGradientBoosting, multiLabelClfDataSet);
        IMLGBJointLabelIsotonicScaling.BucketInfo total = imlgbJointLabelIsotonicScaling.individualProbs(multiLabelClfDataSet);

        double[] counts = total.counts;
        double[] correct = total.sums;
        double[] sumProbs = total.sumProbs;
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
        File outputdir = new File(config.getString("output"));
        outputdir.mkdirs();
        File accsFile = new File(outputdir, "labelIsotonicCalibrationResult.txt");
        FileUtils.writeStringToFile(accsFile, sb.toString());
    }

}