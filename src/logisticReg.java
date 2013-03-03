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
import com.parallax.pipeline.ValueScalingPipe;
import com.parallax.pipeline.VectorNormalizingPipe;
import com.parallax.pipeline.projection.DataNormalizationPipe;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.ml.util.option.*;

// a simple logistic regression class to train and test on the spambase dataset
public class logisticReg {

	static String datadir = "/Users/spchopra/research/ml/dsi/data/";
	static String file = datadir + "spambase.csv";
	static int inpDim = 57;
	static Map<String, String> labelMap = Maps.newHashMap();
	static int nfolds = 10;
	
	
	public static void main(String[] args) {
		// create the label map 
		labelMap.put("1", "" + 1);
		labelMap.put("0", "" + 0);
		
		// create a new datanormalization object
		DataNormalization dnorm = new DataNormalization(inpDim);
		
		// declare and create the data processing pipeline
		Pipeline<File, BinaryClassificationInstance> pipeline;
		pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(",", inpDim, labelMap))
				.addPipe(new DataNormalizationPipe(dnorm))
				.addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
		
		// Iterator<Context<BinaryClassificationInstance>> itt = pipeline.process();
		// System.out.println(itt.next());
		
		// create an instances sink to process the data 
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink(); 
		sink.setSource(pipeline);
		
		// create the binary classification instances 
		BinaryClassificationInstances dataset = sink.next().shuffle();
		
		// print the dataset information
		System.out.println("Number of examples: " + dataset.size());
		System.out.println("Number of negative examples: " + dataset.getNumNeg());
		System.out.println("Number of positive examples: " + dataset.getNumPos());
		
		// build the logistic regression model 
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(inpDim, true);
		Configuration<LogisticRegressionBuilder> co = builder.getConfiguration();
		String[] optString = co.getArgumentsFromOpts();
		for (String str : optString) {
			System.out.println(str);
		}

		OnlineEvaluation eval = new OnlineEvaluation();

		for (int fold = 1; fold <= nfolds; fold++) {
			System.out.println("Training for fold: " + fold);
			LogisticRegression model = builder.build();
			model.train(dataset.getTraining(fold, nfolds));		
			eval.add(dataset.getTesting(fold, nfolds), model);

		}
		System.out.println(eval);
	}
	
}
