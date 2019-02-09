package edu.neu.ccs.pyramid.calibration;


import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;

public class InstanceFeatureExtractorTest {
    public static void main(String[] args) {
        FeatureList featureList = new FeatureList();
        for (int i=0;i<10;i++){
            featureList.add(new Feature());
        }

        InstanceFeatureExtractor instanceFeatureExtractor =new InstanceFeatureExtractor("0-3,5,7-9",featureList);
        System.out.println(instanceFeatureExtractor);

    }

}