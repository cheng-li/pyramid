package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * produce dataset table
 * Created by chengli on 1/25/16.
 */
public class Exp143 {
    static String[] dataNames = {"Scene","RCV1","TMC2007","Mediamill","NUS-WIDE",};
    static int[] dataRows = {1,13, 10,4,7};
    static String[] algorithms = {"BR+LR","BR+Boost","PS+LR","PS+Boost","CC+LR","PCC+LR","ECC+LR","CDN+LR","CRF","CBM+LR","CBM+Boost"};
    static int[] algoC =         {14,     16,        18,     20,        22,     24,      23,      25,       11,   9,       3,};
    static String[] methods = {"BR",      "BR",      "PS",   "PS",      "CC",   "PCC",   "ECC",   "CDN",   "CRF", "CBM",   "CBM"};
    static String[] learners= {"LR",      "GB",      "LR",   "GB",      "LR",   "LR",    "LR",    "LR",    "Linear","LR",   "GB"};
    static String[] impl   =  {"ours",    "ours",    "ours", "ours",    "MEKA", "MEKA",  "MEKA",  "MEKA",  "ours","ours",  "ours"};
    static int domainC = 40;
    static int labelC = 27;
    static int combinatonC = 33;
    static int featureC = 26;
    static int instanceC= 28;

    public static void main(String[] args) throws Exception{
        String[][] table = load();
//        System.out.println(table.length);
//        System.out.println(table[0].length);
//        dataTable(table);
        performance(table);

    }

    static double parse(String string){
        double d= 0;
        if (string.equals("")){
            d= 0;
        } else if (string.equals("NA")){
            d= 0;
        } else {
            d = Double.parseDouble(string);
        }
        return d*100;
    }

    static double parseHam(String string){
        double d= 1;
        if (string.equals("")){
            d= 1;
        } else if (string.equals("NA")){
            d= 1;
        } else {
            d = Double.parseDouble(string);
        }
        return d*100;
    }


    static void performance(String[][] table) throws Exception{
        DecimalFormat df2 = new DecimalFormat("#0.0");
        int numData=dataNames.length;




        int numAlgorithms=algoC.length;
        double[][] accs = new double[numAlgorithms][numData];
        double[][] overlaps = new double[numAlgorithms][numData];
        double[][] hams = new double[numAlgorithms][numData];

        for (int i=0;i<numAlgorithms;i++) {

            for (int j = 0; j < numData; j++) {
                accs[i][j] = parse(table[dataRows[j]][algoC[i]]);
                overlaps[i][j] = parse(table[dataRows[j] + 1][algoC[i]]);
                hams[i][j] = parseHam(table[dataRows[j] + 2][algoC[i]]);
            }
        }

        double[] accMax = new double[numData];
        double[] overMax = new double[numData];
        double[] hamMin = new double[numData];

        for (int d=0;d<numData;d++){
            double m = Double.NEGATIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (accs[a][d]>m){
                    m = accs[a][d];
                }
            }
            accMax[d]=m;
        }

        for (int d=0;d<numData;d++){
            double m = Double.NEGATIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (overlaps[a][d]>m){
                    m = overlaps[a][d];
                }
            }
            overMax[d]=m;
        }

        for (int d=0;d<numData;d++){
            double m = Double.POSITIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (hams[a][d]<m){
                    m = hams[a][d];
                }
            }
            hamMin[d]=m;
        }

//        System.out.println(Arrays.toString(overMax));


        StringBuilder sb = new StringBuilder();
        for (int i=0;i<numAlgorithms;i++){
            if (algorithms[i].equals("CBM+LR")){
                sb.append("\\hline").append("\n");
            }
//            sb.append(algorithms[i]).append(" ");
            sb.append(methods[i]).append("&");
            sb.append(learners[i]).append("&");
            sb.append(impl[i]);
            for (int j=0;j<numData;j++){
                sb.append("&");
                if (accs[i][j]==accMax[j]){
                    sb.append("\\bf{");
                }
                if (accs[i][j]==0){
                    sb.append("---");
                } else {
                    sb.append(df2.format(accs[i][j]));
                }
                if (accs[i][j]==accMax[j]){
                    sb.append("}");
                }
                sb.append("&");
                if (overlaps[i][j] == overMax[j]) {
                    sb.append("\\bf{");
                }
                if (overlaps[i][j]==0){
                    sb.append("---");
                } else {
                    sb.append(df2.format(overlaps[i][j]));
                }

                if (overlaps[i][j] == overMax[j]) {
                    sb.append("}");
                }

            }
            sb.append("\\\\").append("\n");


        }
        sb.append("\\hline").append("\n");
        FileUtils.writeStringToFile(new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/tables/performance.txt"),sb.toString(),false);

    }

    static void performance_hl(String[][] table) throws Exception{
        DecimalFormat df2 = new DecimalFormat("#0.0");
        int numData=dataNames.length;


        

        int numAlgorithms=algoC.length;
        double[][] accs = new double[numAlgorithms][numData];
        double[][] overlaps = new double[numAlgorithms][numData];
        double[][] hams = new double[numAlgorithms][numData];

        for (int i=0;i<numAlgorithms;i++) {

            for (int j = 0; j < numData; j++) {
                accs[i][j] = parse(table[dataRows[j]][algoC[i]]);
                overlaps[i][j] = parse(table[dataRows[j] + 1][algoC[i]]);
                hams[i][j] = parseHam(table[dataRows[j] + 2][algoC[i]]);
            }
        }

        double[] accMax = new double[numData];
        double[] overMax = new double[numData];
        double[] hamMin = new double[numData];

        for (int d=0;d<numData;d++){
            double m = Double.NEGATIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (accs[a][d]>m){
                    m = accs[a][d];
                }
            }
            accMax[d]=m;
        }

        for (int d=0;d<numData;d++){
            double m = Double.NEGATIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (overlaps[a][d]>m){
                    m = overlaps[a][d];
                }
            }
            overMax[d]=m;
        }

        for (int d=0;d<numData;d++){
            double m = Double.POSITIVE_INFINITY;
            for (int a=0;a<numAlgorithms;a++){
                if (hams[a][d]<m){
                    m = hams[a][d];
                }
            }
            hamMin[d]=m;
        }

//        System.out.println(Arrays.toString(overMax));


        StringBuilder sb = new StringBuilder();
        for (int i=0;i<numAlgorithms;i++){
            if (algorithms[i].equals("CBM+LR")){
                sb.append("\\hline").append("\n");
            }
            sb.append(algorithms[i]).append(" ");
            for (int j=0;j<numData;j++){
                sb.append("&");
                if (accs[i][j]==accMax[j]){
                    sb.append("\\bf{");
                }
                if (accs[i][j]==0){
                    sb.append("---");
                } else {
                    sb.append(df2.format(accs[i][j]));
                }
                if (accs[i][j]==accMax[j]){
                    sb.append("}");
                }
                sb.append("&");
                if (overlaps[i][j] == overMax[j]) {
                    sb.append("\\bf{");
                }
                if (overlaps[i][j]==0){
                    sb.append("---");
                } else {
                    sb.append(df2.format(overlaps[i][j]));
                }

                if (overlaps[i][j] == overMax[j]) {
                    sb.append("}");
                }

                sb.append("&");
                if (hams[i][j] == hamMin[j]) {
                    sb.append("\\bf{");
                }
                if (hams[i][j]==100){
                    sb.append("---");
                } else {
                    sb.append(df2.format(hams[i][j]));
                }

                if (hams[i][j] == hamMin[j]) {
                    sb.append("}");
                }
            }
            sb.append("\\\\").append("\n");


        }
        sb.append("\\hline").append("\n");
        FileUtils.writeStringToFile(new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/tables/performance_hl.txt"),sb.toString(),false);

    }

    static void dataTable(String[][] table){


        int numData = dataNames.length;
        StringBuilder sb = new StringBuilder();
        sb.append("\\hline").append("\n");

        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(dataNames[i]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");

        sb.append("domain").append(" ");
        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(table[dataRows[i]][domainC]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");


        sb.append("\\#labels").append(" ");
        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(table[dataRows[i]][labelC]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");

        sb.append("\\#combinations").append(" ");
        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(table[dataRows[i]][combinatonC]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");

        sb.append("\\#features").append(" ");
        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(table[dataRows[i]][featureC]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");

        sb.append("\\#instances").append(" ");
        for (int i=0;i<numData;i++){
            sb.append("&").append(" ").append(table[dataRows[i]][instanceC]).append(" ");
        }
        sb.append("\\\\").append("\n").append("\\hline").append("\n");

        System.out.println(sb.toString());
    }

    static String[][] load() throws Exception{
        File file = new File("/Users/chengli/Downloads/CRF vs MEKA - icml.tsv");
        List<String> lines = FileUtils.readLines(file);
        int numColumns = lines.get(0).split("\t").length;
        int numRows = lines.size();
        String[][] table = new String[numRows][numColumns];
        for (int i=0;i<numRows;i++){
            for (int j=0;j<numColumns;j++){
                table[i][j]="";
            }
        }
        for (int i=0;i<numRows;i++){
            String line = lines.get(i);
            String[] split = line.split("\t");
            for (int j=0;j<split.length;j++){
                table[i][j] = split[j];
            }
        }
        return table;
    }
}
