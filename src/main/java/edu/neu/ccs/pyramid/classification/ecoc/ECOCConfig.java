package edu.neu.ccs.pyramid.classification.ecoc;

/**
 * Created by chengli on 10/5/14.
 */
public class ECOCConfig {
    private CodeMatrix.CodeType codeType = CodeMatrix.CodeType.EXHAUSTIVE;
    private int numFunctions=20;

    public CodeMatrix.CodeType getCodeType() {
        return codeType;
    }

    public ECOCConfig setCodeType(CodeMatrix.CodeType codeType) {
        this.codeType = codeType;
        return this;
    }

    public int getNumFunctions() {
        return numFunctions;
    }

    public ECOCConfig setNumFunctions(int numFunctions) {
        this.numFunctions = numFunctions;
        return this;
    }
}
