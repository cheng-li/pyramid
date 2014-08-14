package edu.neu.ccs.pyramid.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * Created by chengli on 8/7/14.
 */
public class DataSetUtil {
    //todo: keep certain columns of a dataset, also keep settings
    

    static int[] parseStandard(File featureFile, File labelFile, String delimiter)throws Exception{
        int numDataPoints=0;
        try(
                BufferedReader br1 = new BufferedReader((new FileReader(labelFile)));
        ){
            //TODO: make it more robust to empty end lines
            while(br1.readLine()!=null){
                numDataPoints += 1;
            }
        }

        int numFeatures=0;
        try(
                BufferedReader br2 = new BufferedReader(new FileReader(featureFile));
        ){
            String line = br2.readLine();
            numFeatures = line.split(delimiter).length;
        }

        int[] stats = {numDataPoints,numFeatures};
        return stats;
    }

    /**
     * fill up a reg data set
     * @param regDataSet
     * @param featureFile
     * @param labelFile
     * @param delimiter
     * @throws Exception
     */
    static void loadStandard(RegDataSet regDataSet,
                             File featureFile,
                             File labelFile,
                             String delimiter) throws Exception{
        int numFeatures = regDataSet.getNumFeatures();

        try (
                BufferedReader br1 = new BufferedReader(new FileReader(featureFile));
        ){
            int i = 0;
            String line1 = null;
            while((line1=br1.readLine())!=null){
                String[] line1Split = line1.split(delimiter);
                for (int j=0;j<numFeatures;j++){
                    float featureValue = Float.parseFloat(line1Split[j]);
                    regDataSet.setFeatureValue(i, j, featureValue);
                }
                i += 1;
            }
        }

        try (BufferedReader br2 = new BufferedReader((new FileReader(labelFile)));
        ){
            String line2 = null;
            int k = 0;
            while((line2=br2.readLine())!=null){
                String[] line2Split = line2.split(delimiter);
                double label = Double.parseDouble(line2Split[0]);
                regDataSet.setLabel(k, label);
                k += 1;
            }
        }
    }

    /**
     * fill up a clf data set
     * @param clfDataSet
     * @param featureFile
     * @param labelFile
     * @param delimiter
     * @throws Exception
     */
    static void loadStandard(ClfDataSet clfDataSet,
                             File featureFile,
                             File labelFile,
                             String delimiter) throws Exception{
        int numFeatures = clfDataSet.getNumFeatures();

        try (
                BufferedReader br1 = new BufferedReader(new FileReader(featureFile));
        ){
            int i = 0;
            String line1 = null;
            while((line1=br1.readLine())!=null){
                String[] line1Split = line1.split(delimiter);
                for (int j=0;j<numFeatures;j++){
                    float featureValue = Float.parseFloat(line1Split[j]);
                    clfDataSet.setFeatureValue(i,j,featureValue);
                }
                i += 1;
            }
        }

        try (BufferedReader br2 = new BufferedReader((new FileReader(labelFile)));
        ){
            String line2 = null;
            int k = 0;
            while((line2=br2.readLine())!=null){
                String[] line2Split = line2.split(delimiter);
                int label = Integer.parseInt(line2Split[0]);
                clfDataSet.setLabel(k,label);
                k += 1;
            }
        }
    }

}
