package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelSuggester;

import java.io.File;
import java.io.IOException;

/**
 * Created by Rainicy on 9/26/15.
 */
public class Exp209 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String inputData = config.getString("input.data");
        int numClusters = config.getInt("num.clusters");

        MultiLabelClfDataSet multiLabelClfDataSet;
        if (config.getBoolean("isDense")) {
            multiLabelClfDataSet = TRECFormat.loadMultiLabelClfDataSet(
                    new File(inputData), DataSetType.ML_CLF_DENSE, true);
        } else {
            multiLabelClfDataSet = TRECFormat.loadMultiLabelClfDataSet(
                    new File(inputData), DataSetType.ML_CLF_SPARSE, true);
        }

        System.out.println("Starting to train clusters ... ");
        MultiLabelSuggester suggester = new MultiLabelSuggester(multiLabelClfDataSet, numClusters);

        System.out.println("bmm="+suggester.getBmm());

    }
}
