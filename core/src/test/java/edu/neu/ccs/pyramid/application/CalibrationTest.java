package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.stream.IntStream;

public class CalibrationTest{
    public static void main(String[] args) throws Exception{
        CBM cbm = (CBM) Serialization.deserialize("/Users/chengli/tmp/knn_cali/model");
        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/knn_cali/mscoco/valid", DataSetType.ML_CLF_DENSE,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/knn_cali/mscoco/test", DataSetType.ML_CLF_DENSE,true);
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/knn_cali/validPredictions"),pred(cbm,valid));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/knn_cali/testPredictions"),pred(cbm,test));
    }

    private static String pred(CBM cbm, MultiLabelClfDataSet dataSet){
        MultiLabel[] prediction = cbm.predict(dataSet);
        double[] scores = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->cbm.predictAssignmentProb(dataSet.getRow(i),prediction[i]))
                .toArray();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i=0;i<scores.length;i++){
            stringBuilder.append(prediction[i].toSimpleString()).append( "(").append(scores[i]).append(")").append("\n");
        }
        return stringBuilder.toString();
    }

}