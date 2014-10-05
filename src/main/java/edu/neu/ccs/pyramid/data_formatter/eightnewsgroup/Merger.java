package edu.neu.ccs.pyramid.data_formatter.eightnewsgroup;

/**
 * merge 20newsgroup to 8newsgroup
 * Created by chengli on 10/5/14.
 */
public class Merger {
    public static String[] extLabels={
            "religion",
            "computer",
            "forsale",
            "autos",
            "sports",
            "med",
            "space",
            "politics"};

    public static int merged(int oldLabel){
        int label=0;

        if (oldLabel==0||oldLabel==15||oldLabel==19){
            label=0;
        }
        if (oldLabel==1||oldLabel==2||oldLabel==3||oldLabel==4||oldLabel==5||oldLabel==11||oldLabel==12){
            label=1;
        }
        if (oldLabel==6){
            label=2;
        }
        if (oldLabel==7||oldLabel==8){
            label=3;
        }
        if (oldLabel==9||oldLabel==10){
            label=4;
        }
        if (oldLabel==13){
            label=5;
        }
        if (oldLabel==14){
            label=6;
        }
        if (oldLabel==16||oldLabel==17||oldLabel==18){
            label=7;
        }
        return label;
    }

}
