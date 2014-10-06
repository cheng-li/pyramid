package edu.neu.ccs.pyramid.classification.ecoc;



import edu.neu.ccs.pyramid.eval.ConfusionMatrix;
import edu.neu.ccs.pyramid.util.Sampling;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by chengli on 10/4/14.
 */
public class CodeMatrix implements Serializable {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFunctions;
    //stored twice
    private List<boolean[]> codeRows;
    private List<boolean[]> codeColumns;

    public int getNumFunctions() {
        return numFunctions;
    }

    boolean[] getCodeRow(int row){
        return this.codeRows.get(row);
    }

    boolean[] getCodeColumn(int column){
        return this.codeColumns.get(column);
    }


    public static CodeMatrix exhaustiveCodes(int numClasses){
        int numFunctions = (int)Math.pow(2,numClasses-1)-1;
        CodeMatrix codeMatrix = new CodeMatrix(numClasses,numFunctions);
        for (int i=0;i<numClasses;i++){
            fillExhaustive(codeMatrix,i);
        }
        return codeMatrix;
    }

    public static CodeMatrix randomCodes(int numClasses, int numFunctions){
        CodeMatrix codeMatrix = new CodeMatrix(numClasses,numFunctions);
        CodeMatrix exhaustiveCodes = exhaustiveCodes(numClasses);
        int[] sample = Sampling.sampleBySize(exhaustiveCodes.getNumFunctions(),numFunctions);
        for (int i=0;i<numClasses;i++){
            for (int j=0;j<numFunctions;j++){
                int oldColumn = sample[j];
                boolean value = exhaustiveCodes.getCodeRow(i)[oldColumn];
                codeMatrix.setCode(i,j,value);
            }
        }
        return codeMatrix;
    }


    public int[] aggregateLabels(int column, int[] oldLabels){
        boolean[] codeColumn = this.getCodeColumn(column);
        int numDataPoints = oldLabels.length;
        int[] newLabels = new int[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            int oldLabel = oldLabels[i];
            boolean isOne = codeColumn[oldLabel];
            if (isOne){
                newLabels[i] = 1;
            } else {
                newLabels[i] = 0;
            }
        }
        return newLabels;
    }

    /**
     * find best match by distance
     * @param code
     * @return
     */
    public int matchClass(int[] code){
        boolean[] bCode = convertToBoolean(code);
        int matched = 0;
        //all wrong
        int minDis = this.numFunctions;
        for (int i=0;i<this.numClasses;i++){
            int dis = distance(bCode,this.getCodeRow(i));
            if (dis<minDis){
                minDis = dis;
                matched = i;
            }
        }
        return matched;
    }

    //==========PRIVATE==========

    private CodeMatrix(int numClasses, int numFunctions) {
        this.numClasses = numClasses;
        this.numFunctions = numFunctions;
        this.codeRows = new ArrayList<>(numClasses);
        for (int i=0;i<numClasses;i++){
            boolean[] codeRow = new boolean[numFunctions];
            this.codeRows.add(codeRow);
        }

        this.codeColumns = new ArrayList<>(numFunctions);
        for (int j=0;j<numFunctions;j++){
            boolean[] codeColumn = new boolean[numClasses];
            this.codeColumns.add(codeColumn);
        }
    }

    private void setCode(int row, int column, boolean codeValue){
        this.codeRows.get(row)[column] = codeValue;
        this.codeColumns.get(column)[row] = codeValue;
    }

    private static void fillExhaustive(CodeMatrix codeMatrix, int row){
        int numFunctions = codeMatrix.numFunctions;
        int numClasses = codeMatrix.numClasses;
        if (row == 0){
            for(int j=0;j<numFunctions;j++){
                codeMatrix.setCode(0,j,true);
            }
        } else {
            int blockSize = (int)Math.pow(2,numClasses-row-1);
            ExhaustiveFiller(codeMatrix,row,0,blockSize,false);
        }
    }

    private static void ExhaustiveFiller(CodeMatrix codeMatrix, int row,
                                         int start,
                                         int blockSize, boolean startValue){
        int length = codeMatrix.numFunctions;
        if ((start + blockSize-1)>=length-1){
            for (int i=start;i<length;i++){
                codeMatrix.setCode(row, i, startValue);
            }
        } else {
            for (int i=start;i<start+blockSize;i++){
                codeMatrix.setCode(row,i,startValue);
            }
            boolean newValue = !startValue;
            int newStart = start+blockSize;
//            System.out.println("row="+row);
//            System.out.println("start="+start);
            ExhaustiveFiller(codeMatrix,row,newStart,blockSize,newValue);
        }
    }



    private static int distance(boolean[] code1, boolean[] code2){
        int distance = 0;
        for (int i=0;i<code1.length;i++){
            if (code1[i]!=code2[i]){
                distance += 1;
            }
        }
        return distance;
    }

    private static boolean[] convertToBoolean(int[] code){
        boolean[] out = new boolean[code.length];
        for (int i=0;i<code.length;i++){
            if (code[i]==1){
                out[i] = true;
            }
        }
        return out;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<this.numClasses;i++){
            sb.append(Arrays.toString(this.getCodeRow(i))).append("\n");
        }
        return sb.toString();
    }

    public static enum CodeType{
        EXHAUSTIVE,RANDOM
    }
}
