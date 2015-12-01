package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.MLClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

/**
 * Created by chengli on 12/1/15.
 */
public class MultiLabelSynthesizer {

    /**
     * 60: 1,0
     * 40:0,1
     * @return
     */
    public static MultiLabelClfDataSet randomBinary(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(1)
                .numClasses(2)
                .numDataPoints(100)
                .build();

        for (int i=0;i<60;i++){
            dataSet.addLabel(i,0);
        }

        for (int i=60;i<100;i++){
            dataSet.addLabel(i,1);
        }

        return dataSet;
    }


    /**
     * 30: 1,1
     * 40: 1,0
     * 30: 0,1
     * @return
     */
    public static MultiLabelClfDataSet randomTwoLabels(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(1)
                .numClasses(2)
                .numDataPoints(100)
                .build();

        for (int i=0;i<30;i++){
            dataSet.addLabel(i,0);
            dataSet.addLabel(i,1);
        }
        for (int i=30;i<70;i++){
            dataSet.addLabel(i,0);
        }

        for (int i=70;i<100;i++){
            dataSet.addLabel(i,1);
        }

        return dataSet;
    }
}
