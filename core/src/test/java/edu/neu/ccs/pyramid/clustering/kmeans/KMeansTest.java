package edu.neu.ccs.pyramid.clustering.kmeans;


import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class KMeansTest{
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Dropbox/Shared/CS6220DM/2_cluster_EM_mixt/HW2/mnist_features.txt"));

        Collections.shuffle(lines, new Random(0));

        int rows = 100;
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(rows)
                .numFeatures(28*28)
                .build();
        for (int i=0;i<rows;i++){
            String line = lines.get(i);
            String[] split = line.split(" ");
            for (int j=0;j<split.length;j++){
                dataSet.setFeatureValue(i,j,Double.parseDouble(split[j]));
            }
        }

        int numComponents = 10;

        KMeans kMeans = new KMeans(numComponents, dataSet);
        kMeans.randomInitialize();

        for (int k=0;k<numComponents;k++){
            plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/iter_0_component_"+k+".png");
        }






    }



    private static void plot(Vector vector, int height, int width, String imageFile) throws Exception{
        BufferedImage image = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
        for (int i=0;i<width;i++){
            for (int j=0;j<height;j++){
                image.setRGB(j,i,(int)vector.get(i*width+j));
            }
        }
        new File(imageFile).getParentFile().mkdirs();
        ImageIO.write(image,"png",new File(imageFile));
    }



}