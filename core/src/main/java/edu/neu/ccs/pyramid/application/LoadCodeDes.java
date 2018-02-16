package edu.neu.ccs.pyramid.tmp;

import java.io.*;
import java.util.*;

public class LoadCodeDes {
    public static void main(String[] args)throws Exception {
        List<String> label_translator = new ArrayList<>();
        File f = new File("/mnt/home/zhenming/usingCodeDescript/giFeatureMatrix/data_sets/train/label_translator.txt");
        BufferedReader in = new BufferedReader(new FileReader(f));
        String line = in.readLine();
        while(line != null){
            String[] data = line.split("_");
            label_translator.add(data[1]);
            line = in.readLine();
        }
        in.close();
        File g = new File("/mnt/home/zhenming/usingCodeDescript/");
        List<String> codes = new ArrayList<>();
        List<String> descriptions = new ArrayList<>();
        BufferedReader codeDesIn = new BufferedReader(new FileReader(g));
        String sline = codeDesIn.readLine();
        while(sline !=null) {
            String[] sdata = sline.split("\t");
            codes.add(sdata[0]);
            String nsdata = sdata[1].replace("\"", "");
            descriptions.add(nsdata);
            sline = codeDesIn.readLine();
        }
        codeDesIn.close();
        StringBuilder sb = new StringBuilder();
        BufferedWriter bwr = new BufferedWriter(new FileWriter(new File("path")));
        for(String label: label_translator){
            for(int i=0;i<codes.size();i++){
                if(codes.get(i).startsWith(label)){
                    sb.append(descriptions.get(i)).append("\t");
                }
            }
            sb.append("\n");
        }
        bwr.write(sb.toString());
        bwr.close();

        StringBuilder nsb = new StringBuilder();
        BufferedWriter nbwr = new BufferedWriter(new FileWriter(new File("path")));
        for(String label: label_translator){
            for(int i=0;i<codes.size();i++){
                if(codes.get(i).equals(label)){
                    nsb.append(descriptions.get(i)).append("\n");
                }
            }
        }
        nbwr.write(nsb.toString());
        nbwr.close();
    }
}
