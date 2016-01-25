package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * produce dataset table
 * Created by chengli on 1/25/16.
 */
public class Exp143 {
    public static void main(String[] args) throws Exception{
        String[][] table = load();
        System.out.println(table.length);
        System.out.println(table[0].length);
//        dataTable(table);
        performance(table);

    }

    static void performance(String[][] table){
        int numData=5;
        int numAlgorithms=8;
        String[] dataNames = {"scene","emotions","mediamill","NUSWIDE","TMC2007"};
        int[] dataRows = {1,5,7,13,17};
        String[] algorithms = {"CBM+LR","CBM+Boost","CRF","BR+LR","BR+Boost","PS+LR","PS+Boost","CC+LR"};
        int[] algoC = {9,3,11,13,15,16,17,18};
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<numAlgorithms;i++){
            sb.append(algorithms[i]).append(" ");
            for (int j=0;j<numData;j++){
                System.out.println("i="+i+"j="+j);
                System.out.println(""+dataRows[j]+","+algoC[i]);
                sb.append("&").append(table[dataRows[j]][algoC[i]]);
                sb.append("&").append(table[dataRows[j]+1][algoC[i]]);
            }
            sb.append("\\\\").append("\n").append("\\hline").append("\n");
        }
        System.out.println(sb.toString());

    }

    static void dataTable(String[][] table){
        int numData=5;
        String[] dataNames = {"scene","emotions","mediamill","NUSWIDE","TMC2007"};
        int[] dataRows = {1,5,7,13,17};
        int domainC = 35;
        int labelC = 22;
        int combinatonC = 28;
        int featureC = 21;
        int instanceC= 23;

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
