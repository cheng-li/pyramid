package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class CalibrationTest{
    public static void main(String[] args) throws Exception{
        CBM cbm = (CBM) Serialization.deserialize("/Users/chengli/tmp/knn_cali/model");
        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/knn_cali/mscoco/valid", DataSetType.ML_CLF_DENSE,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/knn_cali/mscoco/test", DataSetType.ML_CLF_DENSE,true);
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/knn_cali/validPredictions"),pred(cbm,valid,5));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/knn_cali/testPredictions"),pred(cbm,test,1));
    }

    private static String pred(CBM cbm, MultiLabelClfDataSet dataSet, int top){

        StringBuilder stringBuilder = new StringBuilder();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            double[] marginals = cbm.predictClassProbs(dataSet.getRow(i));
            DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
            BMDistribution bmDistribution = cbm.computeBM(dataSet.getRow(i),0.001);
            for (int k=0;k<top;k++){
                MultiLabel multiLabel = dynamicProgramming.nextHighestVector();
                double score = bmDistribution.logProbability(multiLabel);


                stringBuilder.append(""+i+": "+multiLabel.toSimpleString()).append( " (").append(score).append(")").append("\n");
            }
        }
        return stringBuilder.toString();
    }

}