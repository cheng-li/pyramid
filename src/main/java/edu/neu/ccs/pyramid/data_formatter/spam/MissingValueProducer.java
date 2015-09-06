package edu.neu.ccs.pyramid.data_formatter.spam;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * Created by chengli on 10/21/14.
 */
public class MissingValueProducer {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        double[] percentages = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
        for (double p: percentages){
            produce_train(p);
            produce_test(p);
        }

    }

    private static void produce_train(double p) throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        DataSetUtil.allowMissingValue(dataSet);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                if (Math.random()<p){
                    //todo change back
//                    dataSet.setFeatureValue(i,j,0);
                    dataSet.setFeatureValue(i,j,Double.NaN);
                }
            }
        }
        File folder = new File(TMP,""+p+"_missing");

        TRECFormat.save(dataSet,new File(folder,"train.trec"));
    }

    private static void produce_test(double p) throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        DataSetUtil.allowMissingValue(dataSet);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                if (Math.random()<p){
//                    dataSet.setFeatureValue(i,j,0);
                    dataSet.setFeatureValue(i,j,Double.NaN);
                }
            }
        }
        File folder = new File(TMP,""+p+"_missing");

        TRECFormat.save(dataSet,new File(folder,"test.trec"));
    }
}
