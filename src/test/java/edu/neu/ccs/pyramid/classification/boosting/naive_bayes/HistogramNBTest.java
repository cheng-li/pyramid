package edu.neu.ccs.pyramid.classification.boosting.naive_bayes;

import edu.neu.ccs.pyramid.classification.naive_bayes.Histogram;
import edu.neu.ccs.pyramid.classification.naive_bayes.HistogramNaiveBayes;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * Created by Rainicy on 10/3/14.
 */
public class HistogramNBTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        System.out.println(dataSet.getMetaInfo());

        Histogram histgram = new Histogram(1000);
        System.out.println("Total bins: \t" + histgram.getBins());

        histgram.fit(dataSet.getFeatureColumn(1));

        System.out.println(histgram);

    }

}
