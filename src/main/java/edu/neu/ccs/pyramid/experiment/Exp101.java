package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by chengli on 5/6/15.
 */
public class Exp101 {
//    static String performanceFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/imdb/fixed_test/performance.csv";
    static String seleFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/imdb/fixed_test/selection.csv";


    static String performanceFile = "/Users/chengli/Documents/papers/LTR/Ngrams/CIKM15-Cheng-SkipgramSentiment/figures/amazon_phone/performance.csv";

    public static void main(String[] args) throws Exception{
//        print();
clean();
    }

    public static void print() throws Exception{
        DecimalFormat df = new DecimalFormat("0.00");
        int allColumn = 0;
        String[] h = getHeaders();
        double[][] sel = loadSelection();
        double[][] acc = loadAcc();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <sel.length;i++){
            sb.append(h[i]).append("&");
            sb.append((int)sel[i][allColumn]).append("&");
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

    public static String[] getHeaders(){
        String[] h = {"1&0","2&0","3&0","4&0","5&0","2&1","3&1","4&1","5&1",
                "2&2","3&2","4&2","5&2"};
        return h;
    }

    public static double[][] loadAcc() throws Exception{
        double[][] m = csvToMatrix(performanceFile);
        System.out.println(Arrays.deepToString(m));
        return m;
    }


    public static double[][] loadSelection() throws Exception{
        double[][] m = csvToMatrix(seleFile);
        System.out.println(Arrays.deepToString(m));
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
}
