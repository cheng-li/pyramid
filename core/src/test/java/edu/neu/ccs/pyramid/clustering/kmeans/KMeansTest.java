package edu.neu.ccs.pyramid.clustering.kmeans;


import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class KMeansTest{
    public static void main(String[] args) throws Exception{
//        extractImages();
        test1();

    }

    private static void extractImages() throws Exception{
        FileUtils.cleanDirectory(new File("/Users/chengli/tmp/kmeans_demo"));
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

        for (int i=0;i<rows;i++){
            plot(dataSet.getRow(i), 28,28,
                    "/Users/chengli/tmp/mnist/pic_"+(i+1)+".png");
        }
    }

    private static void test1() throws Exception{
        FileUtils.cleanDirectory(new File("/Users/chengli/tmp/kmeans_demo"));
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
//        kMeans.randomInitialize();
        kMeans.kmeansPlusPlusInitialize();
        boolean showInitialize = true;
        if (showInitialize){
            int[] assignment = kMeans.getAssignments();
            for (int k=0;k<numComponents;k++){
                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/center.png");
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+(k+1)+"_pic_000center.png");

                int counter = 0;
                for (int i=0;i<assignment.length;i++){
                    if (assignment[i]==k){
                        plot(dataSet.getRow(i), 28,28,
                                "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
                        counter+=1;
                    }


//                    if (counter==5){
//                        break;
//                    }
                }
            }
        }
//        for (int k=0;k<numComponents;k++){
//            plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/centers/iter_0_component_"+k+".png");
//            plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+0+"_component_"+k+"_pic_000center.png");
//        }
//
//        int[] assignment0 = kMeans.getAssignments();
//        for (int i=0;i<assignment0.length;i++){
//            plot(dataSet.getRow(i), 28,28,
//                    "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+0+"_component_"+assignment0[i]+"_pic_"+i+".png");
//        }
//
//        System.out.println("objective = "+kMeans.objective());

        for (int iter=1;iter<=5;iter++){
            System.out.println("=====================================");
            System.out.println("iteration "+iter);
            kMeans.iterate();
            System.out.println("objective = "+kMeans.objective());
            int[] assignment = kMeans.getAssignments();
            for (int k=0;k<numComponents;k++){
                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/center.png");
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+(k+1)+"_pic_000center.png");

                int counter = 0;
                for (int i=0;i<assignment.length;i++){
                    if (assignment[i]==k){
                        plot(dataSet.getRow(i), 28,28,
                                "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
                        counter+=1;
                    }


//                    if (counter==5){
//                        break;
//                    }
                }
            }




        }



    }



    private static void test2() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Downloads/fashion-mnist/features.txt"));

        Collections.shuffle(lines, new Random(0));

        int rows = 100;
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(rows)
                .numFeatures(28*28)
                .build();
        for (int i=0;i<rows;i++){
            String line = lines.get(i);
            String[] split = line.split(",");
            for (int j=0;j<split.length;j++){
                dataSet.setFeatureValue(i,j,Double.parseDouble(split[j]));
            }
        }

        int numComponents = 10;

        KMeans kMeans = new KMeans(numComponents, dataSet);
//        kMeans.randomInitialize();
        kMeans.kmeansPlusPlusInitialize();
        for (int k=0;k<numComponents;k++){
            plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/centers/iter_0_component_"+k+".png");
            plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+0+"_component_"+k+"_pic_000center.png");
        }

        int[] assignment0 = kMeans.getAssignments();
        for (int i=0;i<assignment0.length;i++){
            plot(dataSet.getRow(i), 28,28,
                    "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+0+"_component_"+assignment0[i]+"_pic_"+i+".png");
        }

        System.out.println("objective = "+kMeans.objective());

        for (int iter=1;iter<=5;iter++){
            kMeans.iterate();
            System.out.println("objective = "+kMeans.objective());
            for (int k=0;k<numComponents;k++){
                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/centers/iter_"+iter+"_component_"+k+".png");
                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+k+"_pic_000center.png");
            }

            int[] assignment = kMeans.getAssignments();
            for (int i=0;i<assignment.length;i++){
                plot(dataSet.getRow(i), 28,28,
                        "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+assignment[i]+"_pic_"+i+".png");
            }

        }



    }



    private static void plot(Vector vector, int height, int width, String imageFile) throws Exception{

        BufferedImage image = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
//        Graphics2D g2d = image.createGraphics();
//        g2d.setBackground(Color.WHITE);
//
//
//        g2d.fillRect ( 0, 0, image.getWidth(), image.getHeight() );
//        g2d.dispose();
        for (int i=0;i<width;i++){
            for (int j=0;j<height;j++){
                int v = (int)(vector.get(i*width+j));
                int rgb = 65536 * v + 256 * v + v;
                image.setRGB(j,i,rgb);
//                image.setRGB(j,i,(int)(vector.get(i*width+j)/255*16777215));
            }
        }


        new File(imageFile).getParentFile().mkdirs();
        ImageIO.write(image,"png",new File(imageFile));
    }



}