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


    static void performance(String[][] table) throws Exception{
        DecimalFormat df2 = new DecimalFormat("#0.0");
        int numData=5;
        int numAlgorithms=8;
        double[][] accs = new double[numAlgorithms][numData];
        double[][] overlaps = new double[numAlgorithms][numData];


        String[] dataNames = {"scene","emotions","mediamill","NUSWIDE","TMC2007"};
        int[] dataRows = {1,5,7,13,17};
        String[] algorithms = {"BR+LR","BR+Boost","PS+LR","PS+Boost","CC+LR","CRF","CBM+LR","CBM+Boost"};
        int[] algoC = {13,15,17,18,19,11, 9,3,};

        for (int i=0;i<numAlgorithms;i++) {

            for (int j = 0; j < numData; j++) {
                accs[i][j] = parse(table[dataRows[j]][algoC[i]]);
                overlaps[i][j] = parse(table[dataRows[j] + 1][algoC[i]]);
            }
        }

        double[] accMax = new double[numData];
        double[] overMax = new double[numData];
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
//        System.out.println(Arrays.toString(overMax));


        StringBuilder sb = new StringBuilder();
        for (int i=0;i<numAlgorithms;i++){
            sb.append(algorithms[i]).append(" ");
            for (int j=0;j<numData;j++){
                sb.append("&");
                if (accs[i][j]==accMax[j]){
                    sb.append("\\bf{");
                }
                sb.append(df2.format(accs[i][j]));
                if (accs[i][j]==accMax[j]){
                    sb.append("}");
                }
                sb.append("&");
                if (overlaps[i][j] == overMax[j]) {
                    sb.append("\\bf{");
                }
                sb.append(df2.format(overlaps[i][j]));
                if (overlaps[i][j] == overMax[j]) {
                    sb.append("}");
                }
            }
            sb.append("\\\\").append("\n");
            if (algorithms[i].equals("CRF")){
                sb.append("\\hline").append("\n");
            }

        }
        sb.append("\\hline").append("\n");
        FileUtils.writeStringToFile(new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/tables/performance.txt"),sb.toString(),false);

    }

    static void dataTable(String[][] table){
        int numData=5;
        String[] dataNames = {"scene","emotions","mediamill","NUSWIDE","TMC2007"};
        int[] dataRows = {1,5,7,13,17};
        int domainC = 36;
        int labelC = 23;
        int combinatonC = 29;
        int featureC = 22;
        int instanceC= 24;

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
        File file = new File("/Users/chengli/Downloads/table.tsv");
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
