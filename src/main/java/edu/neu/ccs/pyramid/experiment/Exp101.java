package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.stat.inference.TTest;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

/**
 * Created by chengli on 5/6/15.
 */
public class Exp101 {

    //imdb

//    static String performanceFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/imdb/fixed_test/performance.csv";
//    static String seleFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/imdb/fixed_test/selection.csv";
//    static double total = 50000;
//    static double testFrac = 0.5;

    //amazon_phone
    static String folder = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_phone/";
    static String performanceFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_phone/performance.csv";
    static String seleFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_phone/selection.csv";
    static double total = 70211;
    static double testFrac = 0.2;

    //amazon_baby
//    static String folder = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_baby/";
//    static String performanceFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_baby/performance.csv";
//    static String seleFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_baby/selection.csv";
//    static double total = 169411;
//
//    static double testFrac = 0.2;


    public static void main(String[] args) throws Exception{
//        accSelecton();


//        accImprove();
//
//        double[]a = loadLRAccMethod(new File("/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_baby/fold_logs/1"), "elastic");
//        System.out.println(Arrays.toString(a));


//        double[] a= loadSvmAcc(new File("/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_baby/ridge_svm"),11);
//        System.out.println(Arrays.toString(a));

//        System.out.println(prepareFolds());

        accSelectonSig();
    }



    public static void accImprove() throws Exception{
        DecimalFormat df = new DecimalFormat("0.00");
        int allColumn = 0;
        String[] h = getHeaders();
        double[][] acc = loadAcc();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <acc.length;i++){
            if (i<=5||i==10){
                continue;
            }
            sb.append(h[i]).append("&");
            for (int j=0;j<acc[0].length;j++){
                double value = acc[i][j];
                double base = acc[i%5][j];
                double diff = value - base;
                sb.append(df.format(diff)).append("&");
                String portion = ""+(int)(diff*total*testFrac/100);
                sb.append(portion);
                if (j!=acc[0].length-1){
                    sb.append("&");
                }
            }
            sb.append("\\\\");
            sb.append("\n");
            sb.append("\\hline");
            sb.append("\n");
        }

        System.out.println(sb.toString());
    }


    public static void accSelecton() throws Exception{
        DecimalFormat df = new DecimalFormat("0.00");
        int allColumn = 0;
        String[] h = getHeaders();
        double[][] sel = loadSelection();
        double[][] acc = loadAcc();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <sel.length;i++){
            if (i==5 || i==10){
                continue;
            }
            sb.append(h[i]).append("&");
            sb.append(power((int)sel[i][allColumn])).append("&");
            for (int j=0;j<sel[0].length;j++){
                sb.append(df.format(acc[i][j])).append("&");
                String portion = df.format(sel[i][j]/sel[i][allColumn]*100);
                sb.append(portion);
                if (j!=sel[0].length-1){
                    sb.append("&");
                }
            }
            sb.append("\\\\");
            sb.append("\n");
            sb.append("\\hline");
            sb.append("\n");
        }

        System.out.println(sb.toString());
    }


    public static void accSelectonSig() throws Exception{
        DecimalFormat df = new DecimalFormat("0.00");
        int allColumn = 0;
        String[] h = getHeaders();
        double[][] sel = loadSelection();
        double[][] acc = loadAcc();
        double[][] folds = loadFolds();

        System.out.println(Arrays.deepToString(folds));

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <sel.length;i++){
            if (i==5 || i==10){
                continue;
            }
            sb.append(h[i]).append("&");
            sb.append(power((int)sel[i][allColumn])).append("&");
            for (int j=0;j<sel[0].length;j++){
                sb.append(df.format(acc[i][j]));
                if (i>5){
                    double[] currentStat = new double[5];
                    double[] basestats = new double[5];
                    for (int k=0;k<5;k++){
                        currentStat[k] = folds[i][j*5+k];
                        basestats[k] = folds[i%5][j*5+k];

                    }
                    System.out.println("current "+Arrays.toString(currentStat));
                    System.out.println("base "+Arrays.toString(basestats));
                    TTest tTest = new TTest();
                    double p = tTest.pairedTTest(currentStat,basestats);

                    if (p<0.05){
                        sb.append("*");
                    }
                }


                sb.append("&");

                String portion = df.format(sel[i][j]/sel[i][allColumn]*100);
                sb.append(portion);
                if (j!=sel[0].length-1){
                    sb.append("&");
                }
            }
            sb.append("\\\\");
            sb.append("\n");
            sb.append("\\hline");
            sb.append("\n");
        }

        System.out.println(sb.toString());
    }

    public static String[] getHeaders(){
        String[] h = {"1&0","2&0","3&0","4&0","5&0","1&1","2&1","3&1","4&1","5&1","1&2",
                "2&2","3&2","4&2","5&2"};
        return h;
    }

    public static double[][] loadAcc() throws Exception{
        double[][] m = csvToMatrix(performanceFile);
//        System.out.println(Arrays.deepToString(m));
        return m;
    }


    public static double[][] loadSelection() throws Exception{
        double[][] m = csvToMatrix(seleFile);
//        System.out.println(Arrays.deepToString(m));
        return m;
    }

    public static double[][] loadFolds() throws Exception{
        double[][] m = csvToMatrix(new File(folder,"folds.csv").toString());
//        System.out.println(Arrays.deepToString(m));
        return m;
    }


    public static double[][] csvToMatrix(String file)throws Exception{
        List<String> lines = FileUtils.readLines(new File(file));
        int numRow = lines.size();
        int numColumn = lines.get(0).split(",").length;
        double[][] count = new double[numRow][numColumn];
        for (int i=0;i<numRow;i++){
            for (int j=0;j<numColumn;j++){
                count[i][j] = Double.parseDouble(lines.get(i).split(",")[j]);
            }
        }
        return count;
    }

    public static String clean() throws Exception{
        double[][] m = loadAcc();
        for (int i=0;i<m.length;i++){
            for (int j=0;j<m[0].length;j++){
                m[i][j] *= 100;
            }
        }
        StringBuilder sb  = new StringBuilder();
        DecimalFormat df = new DecimalFormat("0.00");
        for (int i=0;i<m.length;i++){
            for (int j=0;j<m[0].length;j++){
               sb.append(df.format(m[i][j]));
                if (j!=m[0].length-1){
                    sb.append(",");
                }
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
        return sb.toString();
    }

    public static String power(int a){
        int power = (int)Math.log10(a);
        int first = a/(int)Math.pow(10,power);
        return "$"+first+"\\times"+10+"^"+power+"$";
    }

    public static double[] loadLRAccMethod(File file, String method) throws Exception{
        List<String> lines = FileUtils.readLines(file);
        double[] arr = new double[5];
        boolean find = false;
        for (String line: lines){
            if (find){
                line = line.replace("[","");
                line = line.replace("]","");
                String[] split = line.split(",");
                for (int i=0;i<5;i++){
                    arr[i] = Double.parseDouble(split[i]);
                }
                return arr;
            }
            if (line.startsWith(method+" accuracy")){
                find = true;
            }
        }
        return arr;
    }

    public static double[] loadSvmAcc(File file, int expNum) throws Exception{
        List<String> lines = FileUtils.readLines(file);
        double[] arr = new double[5];
        int startLine = 0;
        for (int i=0;i<lines.size();i++){
            if (lines.get(i).equals("----"+expNum+"----")){
                startLine = i+1;
                break;
            }
        }
        for (int i=startLine;i<startLine+5;i++){
            String line = lines.get(i);
            int start = line.indexOf("=")+2;
            int end = line.indexOf("%");
            String num = line.substring(start,end);
            arr[i-startLine] = Double.parseDouble(num);
        }
        return arr;
    }

    public static String prepareFolds() throws Exception{
        int[] expNums = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        StringBuilder sb = new StringBuilder();
        for (int i: expNums){
            if (i==4 ||i==7){
                sb.append("\n");
                continue;
            }
            double[] ridge_svm = loadSvmAcc(new File(folder,"ridge_svm"),i);
            for(double a: ridge_svm){
                sb.append(a).append(",");
            }

            double[] lasso_svm = loadSvmAcc(new File(folder,"lasso_svm"),i);
            for(double a: lasso_svm){
                sb.append(a).append(",");
            }
            double[] ridge_lr = loadLRAccMethod(new File(new File(folder,"fold_logs"),""+i),"ridge");
            for(double a: ridge_lr){
                sb.append(a).append(",");
            }

            double[] lasso_lr = loadLRAccMethod(new File(new File(folder,"fold_logs"),""+i),"lasso");
            for(double a: lasso_lr){
                sb.append(a).append(",");
            }

            double[] elastic_lr = loadLRAccMethod(new File(new File(folder,"fold_logs"),""+i),"elastic");
            for (int k=0;k<5;k++){
                sb.append(elastic_lr[k]);
                if (k!=4){
                    sb.append(",");
                }
            }
            sb.append("\n");

        }
        return sb.toString();
    }
}
