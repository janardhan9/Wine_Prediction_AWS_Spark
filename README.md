# Wine Prediction AWS Spark

## Training Model on 1 Master and 3 Workers

```
  spark-submit \
  --class com.mlearning.spark.TrainAndPersistWineQualityRFModel \
  --master spark://<master-ip>:7077 \
  --conf spark.executor.memory=3g \
  --conf spark.driver.memory=3g \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.access.key=access_key \
  --conf spark.hadoop.fs.s3a.secret.key=secret_key \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
  --conf spark.hadoop.fs.s3a.impl.disable.cache=true \
  --packages org.apache.hadoop:hadoop-aws:3.2.0,com.amazonaws:aws-java-sdk-bundle:1.11.375 \
  target/wine-quality-1.0-SNAPSHOT.jar
```

<img width="778" alt="Screenshot 2024-12-09 at 5 55 19 PM" src="https://github.com/user-attachments/assets/71a76f48-9c22-4a3b-bd11-e1b7b3714085">

* Web UI

<img width="1510" alt="Screenshot 2024-12-09 at 6 54 52 PM" src="https://github.com/user-attachments/assets/dfb752b4-4fb8-4b7a-87c0-a81f9aa51882">


## Training Model on Docker Env - https://hub.docker.com/repository/docker/janardhankuruva/wine-predition-aws-spark/tags

```
docker run -e AWS_ACCESS_KEY_ID=access_key -e AWS_SECRET_ACCESS_KEY=secret_key -e AWS_REGION=us-east-1 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models wine-quality
```

<img width="1083" alt="Screenshot 2024-12-09 at 1 32 28 AM" src="https://github.com/user-attachments/assets/b5ade5e5-335e-4835-8b55-942bd3158649">
