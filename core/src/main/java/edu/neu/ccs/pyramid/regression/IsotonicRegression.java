package edu.neu.ccs.pyramid.regression;

public class IsotonicRegression {
    private double[] parameters;

    private double[] fit(double[] a, double[] w){
        double[] aprime = new double[a.length];
        double[] wprime = new double[w.length];
        int[] s = new int[a.length];
        aprime[1]= a[1];
        wprime[1] = w[1];
        int j=1;
        s[0]=0;
        s[1]=1;
        for (int i=2;i<a.length;i++){
            j += 1;
            aprime[j] = a[i];
            wprime[j] = w[i];
            while (j>1 && aprime[j]<aprime[j-1]){
                aprime[j-1] = (wprime[j]*aprime[j]+wprime[j-1]*aprime[j-1])/(wprime[j]+wprime[j-1]);
                wprime[j-1] = wprime[j]+wprime[j-1];
                j -=1;
            }
            s[j] = i;
        }
        double[] parameters = new double[a.length];
        for (int k=1;k<=j;k++){
            for (int l=s[k-1]+1;l<=s[k];l++){
                parameters[l] = aprime[k];
            }
        }
        return parameters;
    }
}
