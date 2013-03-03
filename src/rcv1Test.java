import java.io.File;
import java.util.Map;
import java.util.Iterator;

import com.parallax.pipeline.Context;
import com.google.common.collect.Maps;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.ml.vector.util.ValueScaling;
import com.parallax.ml.util.ScaledNormalizing; 
import com.parallax.ml.projection.DataNormalization;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.text.NumericVWToLabeledVectorPipe;
import com.parallax.pipeline.LabelMappingPipe;
import com.parallax.pipeline.VectorNormalizingPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.ml.util.option.*;
import com.parallax.ml.vector.LinearVector;


// a simple logistic regression class to train and test on the spambase dataset
public class rcv1Test {

	static String datadir = "/Users/spchopra/research/ml/data/";
	static String file = datadir + "rcv1.train.txt";
	
	static int nfolds = 10;
	
	// specify the maximum dimension of the input space 
	static int maxDim = (int) Math.pow(2, 18);
	// specify the label map
	static Map<String, String> labelMap = Maps.newHashMap();
				
	
	public static void main(String[] args) {
		
		// set the values of the labelMap
		labelMap.put("1", "" + 1);
		labelMap.put("-1", "" + 0);
				
		// declare and create the data processing pipeline
		Pipeline<File, BinaryClassificationInstance> pipeline;
		
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericVWToLabeledVectorPipe(maxDim))
				.addPipe(new LabelMappingPipe(labelMap))
				.addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
				
		/**
		Iterator<Context<BinaryClassificationInstance>> itt = pipeline.process();
		for (int i = 0; i < 10; i++){
			System.out.println(itt.next().getData());
		}
		*/
		
		
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink(); 
		sink.setSource(pipeline);
		
		// create the binary classification instances 
		BinaryClassificationInstances dataset = sink.next().shuffle();
		
		// print the dataset information
		System.out.println("Number of examples: " + dataset.size());
		System.out.println("Number of negative examples: " + dataset.getNumNeg());
		System.out.println("Number of positive examples: " + dataset.getNumPos());
		
		// build the logistic regression model 
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(maxDim, true);
		// Configuration<LogisticRegressionBuilder> co = builder.getConfiguration();
		// String[] optString = co.getArgumentsFromOpts();
		// for (String str : optString) {
		//	System.out.println(str);
		// }

		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 1; fold <= nfolds; fold++) {
			long startTime = System.currentTimeMillis();
			System.out.println("Training for fold: " + fold);
			LogisticRegression model = builder.build();
			model.train(dataset.getTraining(fold, nfolds));		
			long endTime = System.currentTimeMillis();
			System.out.println("-- training time: " + (endTime - startTime)/1000 + "s");
			eval.add(dataset.getTesting(fold, nfolds), model);
		}
		System.out.println(eval);
		
	}
	
}
