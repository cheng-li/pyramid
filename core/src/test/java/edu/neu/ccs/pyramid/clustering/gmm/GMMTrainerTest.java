package edu.neu.ccs.pyramid.clustering.gmm;


import edu.neu.ccs.pyramid.clustering.bm.BM;
import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.clustering.kmeans.KMeans;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.stream.IntStream;

public class GMMTrainerTest {
    public static void main(String[] args) throws Exception{
//        test2();
//        fashion();
        spam();
    }

    private static void test2() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/tmp/3gaussian.txt"));
        int dim = lines.get(0).split(" ").length;
        RealMatrix data = new Array2DRowRealMatrix(lines.size(),dim);
        for (int i=0;i<lines.size();i++){
            String[] split = lines.get(i).split(" ");
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j]));
            }
        }


        GMM gmm = new GMM(dim,3, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);

        for (int i=0;i<50;i++){
            trainer.iterate();
            double logLikelihood = IntStream.range(0,data.getRowDimension()).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
        }

        double[][] gammas = trainer.getGammas();
        double[] entropies = IntStream.range(0,data.getRowDimension()).mapToDouble(i->Entropy.entropy(gammas[i])).toArray();
        System.out.println(Arrays.toString(entropies));
        int max = ArgMax.argMax(entropies);

        System.out.println(gmm);
    }


    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/Downloads/scene/train_test_split/train", DataSetType.ML_CLF_DENSE,true);
        RealMatrix data = new Array2DRowRealMatrix(dataSet.getNumDataPoints(),dataSet.getNumFeatures());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                data.setEntry(i,j,dataSet.getRow(i).get(j));
            }
        }

        int numComponents=10;
        GMM gmm = new GMM(dataSet.getNumFeatures(),numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);


        for (int i=1;i<=5;i++){
            System.out.println("iteration = "+i);
            trainer.iterate();
            double logLikelihood = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
            Serialization.serialize(gmm, "/Users/chengli/tmp/gmm/model_iter_"+i);
            for (int k=0;k<gmm.getNumComponents();k++){
                FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/mean_iter_"+i+"_component_"+(k+1)),
                        gmm.getGaussianDistributions()[k].getMean().toString().replace("{","")
                                .replace("}","").replace(";",","));
            }
        }


    }

    private static void test3() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Downloads/fashion-mnist/features.txt"));
//        Collections.shuffle(lines);
        int dim = 784;
        int rows = 100;
        RealMatrix data = new Array2DRowRealMatrix(rows,dim);
        for (int i=0;i<rows;i++){
            String[] split = lines.get(i).split(",");
            System.out.println(Arrays.toString(split));
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j])+Math.random());
            }
        }




        int numComponents = 10;
        GMM gmm = new GMM(dim,numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);


        for (int i=1;i<=5;i++){
            System.out.println("iteration = "+i);
            trainer.iterate();
            double logLikelihood = IntStream.range(0,rows).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
            Serialization.serialize(gmm, "/Users/chengli/tmp/gmm/model_iter_"+i);
            for (int k=0;k<gmm.getNumComponents();k++){
                FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/mean_iter_"+i+"_component_"+(k+1)),
                        gmm.getGaussianDistributions()[k].getMean().toString().replace("{","")
                                .replace("}","").replace(";",","));
            }
        }




    }


    private static void fashion() throws Exception{
        FileUtils.cleanDirectory(new File("/Users/chengli/tmp/kmeans_demo"));
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Dropbox/Shared/CS6220DM/data/fashion/features.txt"));

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
                dataSet.setFeatureValue(i,j,Double.parseDouble(split[j])/255);
            }
        }

        int numComponents = 3;

//        KMeans kMeans = new KMeans(numComponents, dataSet);
////        kMeans.randomInitialize();
//        kMeans.kmeansPlusPlusInitialize(100);
//        List<Double> objectives = new ArrayList<>();
//        boolean showInitialize = true;
//        if (showInitialize){
//            int[] assignment = kMeans.getAssignments();
//            for (int k=0;k<numComponents;k++){
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/center.png");
////                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+(k+1)+"_pic_000center.png");
//
//                int counter = 0;
//                for (int i=0;i<assignment.length;i++){
//                    if (assignment[i]==k){
//                        plot(dataSet.getRow(i), 28,28,
//                                "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
//                        counter+=1;
//                    }
//
//                }
//            }
//        }
//        objectives.add(kMeans.objective());
//
//
//        for (int iter=1;iter<=5;iter++){
//            System.out.println("=====================================");
//            System.out.println("iteration "+iter);
//            kMeans.iterate();
//            objectives.add(kMeans.objective());
//            int[] assignment = kMeans.getAssignments();
//            for (int k=0;k<numComponents;k++){
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/center.png");
//                for (int i=0;i<assignment.length;i++){
//                    if (assignment[i]==k){
//                        plot(dataSet.getRow(i), 28,28,
//                                "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
//                    }
//                }
//            }
//
//            System.out.println("training objective changes: "+objectives);
//        }


//        int[] assignments = kMeans.getAssignments();
        RealMatrix data = new Array2DRowRealMatrix(rows,dataSet.getNumFeatures());
        for (int i=0;i<rows;i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                data.setEntry(i,j,dataSet.getRow(i).get(j));
            }
        }


        GMM gmm = new GMM(dataSet.getNumFeatures(),numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);
//        double[][] gammas = new double[assignments.length][numComponents];
//        for (int i=0;i<assignments.length;i++){
//            gammas[i][assignments[i]]=1;
//        }

//        trainer.setGammas(gammas);
        System.out.println("start training GMM");
        for (int i=1;i<=5;i++){
//            trainer.mStep();
//            trainer.eStep();
            trainer.iterate();
            double[][] gammas = trainer.getGammas();
            System.out.println(Arrays.toString(gammas[0]));
            System.out.println(Arrays.toString(gammas[1]));
//            double[] entropies = IntStream.range(0,rows).mapToDouble(i->Entropy.entropy(gammas[i])).toArray();
//            System.out.println(Arrays.toString(entropies));
//            int max = ArgMax.argMax(entropies);
//            plot(dataSet.getRow(max), 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/max_entropy.png");
//            System.out.println(Arrays.toString(gammas[max]));
            double logLikelihood = IntStream.range(0,rows).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);

            for (int k=0;k<numComponents;k++){
                plot(gmm.getGaussianDistributions()[k].getMean(), 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+i+"/cluster_"+(k+1)+"/center.png");
//                for (int i=0;i<assignment.length;i++){
//                    if (assignment[i]==k){
//                        plot(dataSet.getRow(i), 28,28,
//                                "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
//                    }
//                }
            }
        }

//        double[][] gammas = trainer.getGammas();
//        double[] entropies = IntStream.range(0,rows).mapToDouble(i->Entropy.entropy(gammas[i])).toArray();
//        System.out.println(Arrays.toString(entropies));
//        int max = ArgMax.argMax(entropies);
//        plot(dataSet.getRow(max), 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/max_entropy.png");
//        System.out.println(Arrays.toString(gammas[max]));

//        System.out.println(gmm);

    }



    private static void spam() throws Exception{
        FileUtils.cleanDirectory(new File("/Users/chengli/tmp/kmeans_demo"));


        DataSet dataSet = TRECFormat.loadClfDataSet("/Users/chengli/tmp/spam/train",DataSetType.CLF_DENSE,true);
//        DataSet dataSet = TRECFormat.loadRegDataSet("/Users/chengli/tmp/housing/train",DataSetType.REG_DENSE,true);

        int numComponents = 5;

//        KMeans kMeans = new KMeans(numComponents, dataSet);
////        kMeans.randomInitialize();
//        kMeans.kmeansPlusPlusInitialize(100);
//        List<Double> objectives = new ArrayList<>();
//        boolean showInitialize = true;
//        if (showInitialize){
//            int[] assignment = kMeans.getAssignments();
//            for (int k=0;k<numComponents;k++){
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/center.png");
////                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"_component_"+(k+1)+"_pic_000center.png");
//
//                int counter = 0;
//                for (int i=0;i<assignment.length;i++){
//                    if (assignment[i]==k){
//                        plot(dataSet.getRow(i), 28,28,
//                                "/Users/chengli/tmp/kmeans_demo/clusters/initial/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
//                        counter+=1;
//                    }
//
//                }
//            }
//        }
//        objectives.add(kMeans.objective());
//
//
//        for (int iter=1;iter<=5;iter++){
//            System.out.println("=====================================");
//            System.out.println("iteration "+iter);
//            kMeans.iterate();
//            objectives.add(kMeans.objective());
//            int[] assignment = kMeans.getAssignments();
//            for (int k=0;k<numComponents;k++){
//                plot(kMeans.getCenters()[k], 28,28, "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/center.png");
//                for (int i=0;i<assignment.length;i++){
//                    if (assignment[i]==k){
//                        plot(dataSet.getRow(i), 28,28,
//                                "/Users/chengli/tmp/kmeans_demo/clusters/iter_"+iter+"/cluster_"+(k+1)+"/pic_"+(i+1)+".png");
//                    }
//                }
//            }
//
//            System.out.println("training objective changes: "+objectives);
//        }


//        int[] assignments = kMeans.getAssignments();
        RealMatrix data = new Array2DRowRealMatrix(dataSet.getNumDataPoints(),dataSet.getNumFeatures());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                data.setEntry(i,j,dataSet.getRow(i).get(j));
            }
        }


        GMM gmm = new GMM(dataSet.getNumFeatures(),numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);
//        double[][] gammas = new double[assignments.length][numComponents];
//        for (int i=0;i<assignments.length;i++){
//            gammas[i][assignments[i]]=1;
//        }

//        trainer.setGammas(gammas);
        System.out.println("start training GMM");
        for (int i=1;i<=50;i++){
//            trainer.mStep();
//            trainer.eStep();
            trainer.iterate();
            double logLikelihood = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
        }

        double[][] gammas = trainer.getGammas();
        double[] entropies = IntStream.range(0,dataSet.getNumDataPoints()).mapToDouble(i->Entropy.entropy(gammas[i])).toArray();
        System.out.println(Arrays.toString(entropies));
        int max = ArgMax.argMax(entropies);

        System.out.println(Arrays.toString(gammas[max]));

//        System.out.println(gmm);

    }


    private static void plot(RealVector vector, int height, int width, String imageFile) throws Exception{

        BufferedImage image = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
//        Graphics2D g2d = image.createGraphics();
//        g2d.setBackground(Color.WHITE);
//
//
//        g2d.fillRect ( 0, 0, image.getWidth(), image.getHeight() );
//        g2d.dispose();
        for (int i=0;i<width;i++){
            for (int j=0;j<height;j++){
                int v = (int)(vector.getEntry(i*width+j));
                int rgb = 65536 * v + 256 * v + v;
                image.setRGB(j,i,rgb);
//                image.setRGB(j,i,(int)(vector.get(i*width+j)/255*16777215));
            }
        }


        new File(imageFile).getParentFile().mkdirs();
        ImageIO.write(image,"png",new File(imageFile));
    }


    private static void plot(org.apache.mahout.math.Vector vector, int height, int width, String imageFile) throws Exception{

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