package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.IOException;

/**
 * streaming ngram feature selection by regression stump
 * Created by chengli on 2/7/17.
 */
public class StumpSelector {
    /**
     *
     * @param index
     * @param labels size = num labels * num data
     * @param feature
     * @param idTranslator
     * @param matchScoreType
     * @param docFilter
     */
    public static double[] scores(ESIndex index, double[][] labels,
                             Ngram feature,
                             IdTranslator idTranslator, FeatureLoader.MatchScoreType matchScoreType, String docFilter){
        Ngram ngram = null;
        try {
            ngram = (Ngram) Serialization.deepCopy(feature);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        ngram.setIndex(0);
        
        DataSet dataSet = new SparseDataSet(labels[0].length,1, false, null);
        FeatureLoader.loadNgramFeature(index, dataSet, ngram, idTranslator, matchScoreType, docFilter);
        double[] scores = new double[labels.length];
        for (int l=0;l<scores.length;l++){
            double score = score(dataSet, labels[l]);
            scores[l] = score;
        }
        return scores;
    }

    private static double score(DataSet dataSet, double[] labels){
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(2);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig, dataSet, labels);
        return tree.getRoot().getReduction();
    }
}
