package com.mlearning.spark;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

/**
 * This class handles training a Logistic Regression model using Spark,
 * evaluating it, and persisting the model to AWS S3.
 */
public class TrainAndPersistWineQualityDataModel {

    public static final Logger logger = LogManager.getLogger(TrainAndPersistWineQualityDataModel.class);

    // Define S3 paths (replace with your actual bucket and paths)
    private static final String TRAINING_DATASET = "s3a://wine-dataset-spark/Training_Dataset.csv";
    private static final String VALIDATION_DATASET = "s3a://wine-dataset-spark/Validation_Dataset.csv";
    private static final String MODEL_PATH = "models/RandomForestModel/";
    private static final String PREDICTIONS_PATH = "data/predictions/";

    public static void main(String[] args) {
        // Set logging levels to reduce verbosity
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.ERROR);

        // Initialize SparkSession with updated AWS configuration
        SparkSession spark = SparkSession.builder()
                .appName("Wine-Quality-Training")
                .master("local[*]")
                .config("spark.hadoop.fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
                .config("spark.hadoop.fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
                .config("spark.hadoop.fs.s3a.endpoint", "s3." + System.getenv("AWS_REGION") + ".amazonaws.com")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                       "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .getOrCreate();

        TrainAndPersistWineQualityDataModel trainer = new TrainAndPersistWineQualityDataModel();
        trainer.randomForestClassification(spark);
    }

    /**
     * Trains a Logistic Regression model using the training dataset,
     * evaluates it on the validation dataset, and saves the model to S3.
     *
     * @param spark The SparkSession object.
     */
    public void logisticRegression(SparkSession spark) {
        logger.info("Loading training data from: " + TRAINING_DATASET);
        Dataset<Row> trainingDf = getDataFrame(spark, true, TRAINING_DATASET).cache();

        logger.info("Training Logistic Regression model");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.0)
                .setElasticNetParam(0.8);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{lr});
        PipelineModel model = pipeline.fit(trainingDf);

        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model.stages()[0]);
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();

        double accuracy = trainingSummary.accuracy();
        double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
        double truePositiveRate = trainingSummary.weightedTruePositiveRate();
        double fMeasure = trainingSummary.weightedFMeasure();
        double precision = trainingSummary.weightedPrecision();
        double recall = trainingSummary.weightedRecall();

        System.out.println();
        System.out.println("Training DataSet Metrics ");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("FPR: " + falsePositiveRate);
        System.out.println("TPR: " + truePositiveRate);
        System.out.println("F-measure: " + fMeasure);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);

        logger.info("Loading validation data from: " + VALIDATION_DATASET);
        Dataset<Row> validationDf = getDataFrame(spark, true, VALIDATION_DATASET).cache();

        logger.info("Transforming validation data with the trained model");
        Dataset<Row> validationPredictions = model.transform(validationDf).cache();

        System.out.println("\nValidation DataSet Metrics:");
        validationPredictions.select("features", "label", "prediction").show(5, false);
        printMetrics(validationPredictions);

        // Save the model and predictions locally
        try {
            logger.info("Saving predictions to: " + PREDICTIONS_PATH);
            validationPredictions.write()
                .mode("overwrite")
                .parquet(PREDICTIONS_PATH);
            logger.info("Predictions saved successfully.");

            logger.info("Saving the trained model to: " + MODEL_PATH);
            model.write().overwrite().save(MODEL_PATH);
            logger.info("Model saved successfully.");
        } catch (IOException e) {
            logger.error("Failed to save the model or predictions: ", e);
            e.printStackTrace();
        }
    }

    /**
     * Trains a Random Forest model using the training dataset,
     * evaluates it on the validation dataset, and saves the model to S3.
     *
     * @param spark The SparkSession object.
     */
    public void randomForestClassification(SparkSession spark) {
        logger.info("Loading training data from: " + TRAINING_DATASET);
        Dataset<Row> trainingDf = getDataFrame(spark, true, TRAINING_DATASET).cache();

        logger.info("Training Random Forest model");
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(10)
                .setSeed(42);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{rf});
        PipelineModel model = pipeline.fit(trainingDf);

        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) (model.stages()[0]);

        // Print feature importances
        System.out.println("Feature Importances:");
        for (int i = 0; i < rfModel.featureImportances().size(); i++) {
            System.out.println("Feature " + i + " importance: " + rfModel.featureImportances().toArray()[i]);
        }

        logger.info("Loading validation data from: " + VALIDATION_DATASET);
        Dataset<Row> validationDf = getDataFrame(spark, true, VALIDATION_DATASET).cache();
        Dataset<Row> validationPredictions = model.transform(validationDf).cache();

        System.out.println("\nValidation DataSet Metrics:");
        validationPredictions.select("features", "label", "prediction").show(5, false);
        printMetrics(validationPredictions);

        // Save model and predictions
        try {
            logger.info("Saving predictions to: " + PREDICTIONS_PATH);
            validationPredictions.write()
                .mode("overwrite")
                .parquet(PREDICTIONS_PATH);
            logger.info("Predictions saved successfully.");

            logger.info("Saving the trained model to: " + MODEL_PATH);
            model.write().overwrite().save(MODEL_PATH);
            logger.info("Model saved successfully.");
        } catch (IOException e) {
            logger.error("Failed to save the model or predictions: ", e);
            e.printStackTrace();
        }
    }

    /**
     * Evaluates and prints various classification metrics.
     *
     * @param predictions The Dataset containing predictions and labels.
     */
    public void printMetrics(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Weighted Precision: " + precision);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Weighted Recall: " + recall);
    }

    /**
     * Reads a CSV file from the specified path, assembles features, and returns the prepared Dataset.
     *
     * @param spark     The SparkSession object.
     * @param transform Whether to transform the data using VectorAssembler.
     * @param path      The S3 path to the CSV file.
     * @return The prepared Dataset<Row>.
     */
    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String path) {
        logger.info("Reading data from: " + path);
        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                //.option("sep", ";") // Uncomment if CSV uses semicolon separator
                //.option("quote", "\"") // Uncomment if needed
                .load(path);

        // Rename columns if necessary
        Dataset<Row> lblFeatureDf = df.withColumnRenamed("quality", "label")
                .select("label", "alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide", "chlorides",
                        "residual sugar", "citric acid", "volatile acidity", "fixed acidity");

        lblFeatureDf = lblFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide", "chlorides",
                        "residual sugar", "citric acid", "volatile acidity", "fixed acidity"})
                .setOutputCol("features");

        if (transform) {
            lblFeatureDf = assembler.transform(lblFeatureDf).select("label", "features");
        }

        return lblFeatureDf;
    }
}