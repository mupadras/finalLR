package edu.indiana.dsc.sparkmllr;

/**
 * Created by madhu on 7/20/16.
 */
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import Jama.*;

public class SparkWeightInterceptLR {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setAppName("Linear Regression with Elastic Net Example");

        SparkContext sc = new SparkContext(conf);
        SQLContext sql = new SQLContext(sc);
        String path = "/Users/madhu/Desktop/sample_libsvm_data.txt";

        // Load training data
        DataFrame training = sql.createDataFrame(MLUtils.loadLibSVMFile(sc, path).toJavaRDD(), LabeledPoint.class);

        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Fit the model
        LinearRegressionModel lrModel = lr.fit(training);

        // Print the weights and intercept for linear regression
        System.out.println("Weights: " + lrModel.weights() + " Intercept: " + lrModel.intercept());

        // Summarize the model over the training set and print out some metrics
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());
    }
}


