package solver;

import java.util.Arrays;

import math.Matrix;
import math.Vector;

/**
 * 
 * NOTE POUR MOI-MEME APRES LE RAGNAROK : changer lookstep min (pour 12 actions)
 * -> A* pathfinding BATCH NORM COMPLÉTÉ. IMPLÉMENTER DROPOUT SI BESOIN IL Y A.
 * 
 * @author clawrent
 *
 */
public class NeuralNet {

	/** Should be at least 10_000 for real improvment. 60_000 normally **/
	private static int TRAINING_ITERATIONS = 1;// 30000

	/** Number of hidden layers **/
	private static int HIDDEN_LAYERS_NUMBER = 3;

	/** Application : all hidden layers **/
	private static int HIDDEN_LAYERS_SIZE = 16; // 500

	/**
	 * Application : all layers<br>
	 * Optimal number between 0.6f and 0.0001f
	 **/
	public static double LEARNING_RATE = 1e-2f;

	/** BatchNorm Gamma Parameter learning rate */
	public static double GAMMA_LEARNING_RATE = 1e-2f;

	/** BatchNorm Biases Parameter learning rate */
	public static double BETA_LEARNING_RATE = 1e-2f;

	/** BatchNorm Numerical stability : 1f/(Math.sqrt(EPSILON)) **/
	private static float EPSILON = 1e-8f;

	/** BatchNorm Running mean learning rate **/
	private static float MOMENTUM = 0.9f;

	/** Number of neurons for 1 row of training input set **/
	private int inputWidth;
	/** Number of neurons for 1 row of training output set **/
	private int outputWidth;

	/** FLAG for running mean **/
	private boolean training;

	private int iteration;
	private float[][] batchRunningMean;
	private float[][] batchRunningVariance;

	private int batchSize;
	private int batchIndex;

	private float[][][] inputBatch;
	private float[][][] outputBatch;
	private float[][][] resultBatch;

	private float[][] inputLayer;
	private float[][] outputLayer;
	private float[][] resultLayer;

	private float[][][] synapses;

	private float[][][] hiddenLayers;

	/**
	 * Learnable BatchNorm parameter for each layer x<br>
	 * apply transformation : gamma[x] * layer[x] + beta[x].
	 **/
	private float[][] batchGamma, batchBeta;

	/** BatchNorm propagation cache variable. Used later in backpropagation. **/
	private float[][][][] batchXFixedMean;
	/** BatchNorm propagation cache variable. Used later in backpropagation. **/
	private float[][][] batchSqrtInvVariance;

	/** BatchNorm propagation cache variable. Used later in backpropagation. **/
	private float[][][][] batchLayerNorm;

	/**
	 * Momentum SGD
	 */
	private float[][][] sgdRunningGradient;
	private float[][][] rmsRunningGradient;

	/**
	 * Mean squared error : net.getMSE()
	 */
	private float mse;

	public double sigmoid(double x) {
		return (1.0 / (1.0 + Math.exp(-x)));
	}

	public double sigmoidInverse(double x) {
		return x * (1.0 - x);
	}

	public double tanh(double x) {
		double e = Math.exp(x);
		double inv = 1 / e;
		return (e - inv) / (e + inv);
	}

	public double tanhInverse(double x) {
		return 1 - x * x;
	}

	public double relu(double x) {
		return Math.max(0, x);
	}

	public double reluInverse(double x) {
		return (x > 0 ? 1 : 0);
	}

	public float leakyRelu(float x) {
		return Math.max(0.01f * x, x);
	}

	public float leakyReluInverse(float x) {
		return x > 0 ? 1f : 0.01f;
	}

	public float[][] activationFunction(float[][] layer) {
		return Matrix.apply(layer, x -> leakyRelu(x));
	}

	public float[][] backpropagationFunction(float[][] layer) {
		return Matrix.apply(layer, x -> leakyReluInverse(x));
	}

	public float[][] resultActivationFunction(float[][] layer) {
		return activationFunction(layer);// Matrix.apply(layer, x ->
											// sigmoid(x));
	}

	public float[][] resultBackpropagationFunction(float[][] layer) {
		return backpropagationFunction(layer);// Matrix.apply(layer, x ->
												// sigmoidInverse(x));
	}

	public float weightRandomValue(int pFanIn, int pFanOut) {
		return (float) ((2 * Math.random() - 1) * Math.sqrt(1d / ((pFanOut + pFanIn))) *1e-9d) ;
	}

	public float[][] lossFunction(float[][] output, float[][] result) {
		// float s = Matrix.mean(Math.pow(me1-me2, 2))));
		// return Matrix.build(output.length, output[0].length, (i,j)->s);

		return Matrix.substraction(output, result);// Matrix.multiply(, 0.001f);
	}

	/**
	 * Initialize a neural network instance. inputLayer and outputLayer parameters
	 * must first be set to ensure training.
	 * 
	 * @param inWidth  Number of neurons for 1 row of training input set
	 * @param outWidth Number of neurons for 1 row of training output set
	 */
	public NeuralNet(int inWidth, int outWidth) {
		inputWidth = inWidth;
		outputWidth = outWidth;

		inputLayer = null;
		outputLayer = null;

		training = false;

		mse = 0f;
		iteration = 0;

		synapses = new float[HIDDEN_LAYERS_NUMBER + 1][][];
		sgdRunningGradient = new float[HIDDEN_LAYERS_NUMBER + 1][][];
		rmsRunningGradient = new float[HIDDEN_LAYERS_NUMBER + 1][][];
		if (HIDDEN_LAYERS_NUMBER == 0) {
			synapses[0] = Matrix.build(inputWidth, outputWidth, (i, j) -> weightRandomValue(inputWidth, outputWidth));
			sgdRunningGradient[0] = new float[inputWidth][outputWidth];
			rmsRunningGradient[0] = new float[inputWidth][outputWidth];
		} else {
			hiddenLayers = new float[HIDDEN_LAYERS_NUMBER][][];

			synapses[0] = Matrix.build(inputWidth, HIDDEN_LAYERS_SIZE,
					(i, j) -> weightRandomValue(inputWidth, HIDDEN_LAYERS_SIZE));
			sgdRunningGradient[0] = new float[inputWidth][HIDDEN_LAYERS_SIZE];
			rmsRunningGradient[0] = new float[inputWidth][HIDDEN_LAYERS_SIZE];
			
			synapses[HIDDEN_LAYERS_NUMBER] = Matrix.build(HIDDEN_LAYERS_SIZE, outputWidth,
					(i, j) -> weightRandomValue(HIDDEN_LAYERS_SIZE, outputWidth));
			sgdRunningGradient[HIDDEN_LAYERS_NUMBER] = new float[HIDDEN_LAYERS_SIZE][outputWidth];
			rmsRunningGradient[HIDDEN_LAYERS_NUMBER] = new float[HIDDEN_LAYERS_SIZE][outputWidth];
			
			for (int k = 1; k < HIDDEN_LAYERS_NUMBER; k++) {
				synapses[k] = Matrix.build(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE,
						(i, j) -> weightRandomValue(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE));
				sgdRunningGradient[k] = new float[HIDDEN_LAYERS_SIZE][HIDDEN_LAYERS_SIZE];
				rmsRunningGradient[k] = new float[HIDDEN_LAYERS_SIZE][HIDDEN_LAYERS_SIZE];
			}
		}
		// reset batch norm
		initBatchNormParameters();
		resetRunningGradient();
	}

	public void initBatchNormParameters() {
		/// batch Norm reset
		batchGamma = new float[HIDDEN_LAYERS_NUMBER][HIDDEN_LAYERS_SIZE];
		batchBeta = new float[HIDDEN_LAYERS_NUMBER][HIDDEN_LAYERS_SIZE];

		batchRunningMean = new float[HIDDEN_LAYERS_NUMBER][HIDDEN_LAYERS_SIZE];
		batchRunningVariance = new float[HIDDEN_LAYERS_NUMBER][HIDDEN_LAYERS_SIZE];

		for (int i = 0; i < HIDDEN_LAYERS_NUMBER; i++) {
			for (int n = 0; n < HIDDEN_LAYERS_SIZE; n++) {
				batchGamma[i][n] = 1;
				batchBeta[i][n] = 0;
				batchRunningVariance[i][n] = 1;
				batchRunningMean[i][n] = 0;
			}
		}
	}

	public void setBatchSize(int pBatchSize) {
		if (pBatchSize > 0)
			batchSize = pBatchSize;
	}

	public void setInputLayer(float[][] mInput) {
		if (batchSize == 0)
			throw new IllegalArgumentException("Must first define batchSize!");
		if (mInput[0].length != inputWidth)
			throw new IllegalArgumentException(
					"Given input layer does not have the same number of neurons as its predecessor");

		inputBatch = Matrix.toBatch(batchSize, mInput);
		inputLayer = inputBatch[0];

		batchIndex = 0;

	}

	public void setOutputLayer(float[][] mOutput) {
		if (batchSize == 0)
			throw new IllegalArgumentException("Must first define batchSize!");
		if (mOutput[0].length != outputWidth)
			throw new IllegalArgumentException(
					"Given output layer does not have the same number of neurons as its predecessor");

		outputBatch = Matrix.toBatch(batchSize, mOutput);
		outputLayer = outputBatch[0];

		resultBatch = new float[outputBatch.length][][];

		/// reset sgd running gradient average
		resetRunningGradient();

	}

	private void resetRunningGradient() {
		for (int i = 0; i < HIDDEN_LAYERS_NUMBER + 1; i++) {
			sgdRunningGradient[i] = new float[sgdRunningGradient[i].length][sgdRunningGradient[i][0].length];
			rmsRunningGradient[i] = new float[rmsRunningGradient[i].length][rmsRunningGradient[i][0].length];
		}
	}
	/**
	 * The resulting layer after using {@link propagation()}
	 * 
	 * @return the ouput layer
	 */
	public float[][] getResultLayer() {
		return resultLayer;
	}

	/**
	 * Apply the batch norm feedforward transformation.
	 * 
	 * @param layerIndex needed to cache mean and variance values
	 * @param mInput     the current layer
	 * @return
	 */
	public float[][] batchNormProp(int layerIndex, float[][] mInput) {
		float[][] r = null;

		float[] mean;
		float[] variance;

		if (training) {
			mean = Matrix.colmean(mInput);

			variance = Matrix.colmean(Matrix.square(Matrix.substraction(mInput, mean)));

			// Must be per layer
			batchRunningMean[layerIndex] = Vector.addition(Vector.multiply(batchRunningMean[layerIndex], MOMENTUM),
					Vector.multiply(mean, 1f - MOMENTUM));
			// System.out.println(Arrays.toString(batchRunningMean));
			float adjust = batchSize / Math.max(batchSize - 1f, 1);
			batchRunningVariance[layerIndex] = Vector.addition(
					Vector.multiply(batchRunningVariance[layerIndex], MOMENTUM),
					Vector.multiply(variance, (1f - MOMENTUM) * adjust));

			batchXFixedMean[batchIndex][layerIndex] = Matrix.substraction(mInput, mean);
			batchSqrtInvVariance[batchIndex][layerIndex] = Vector.sqrtinv(variance, EPSILON);
			batchLayerNorm[batchIndex][layerIndex] = Matrix.multiplication(batchXFixedMean[batchIndex][layerIndex],
					batchSqrtInvVariance[batchIndex][layerIndex]);

			r = Matrix.multiplication(batchLayerNorm[batchIndex][layerIndex], batchGamma[layerIndex]);
			r = Matrix.addition(r, batchBeta[layerIndex]);
		} else {

			mean = batchRunningMean[layerIndex];
			variance = batchRunningVariance[layerIndex];

			float[] d = Vector.division(batchGamma[layerIndex], Vector.sqrt(batchRunningVariance[layerIndex], EPSILON));
			float[] t = Vector.multiplication(d, batchRunningMean[layerIndex]);
			r = Matrix.multiplication(mInput, d);
			r = Matrix.addition(r, Vector.substraction(batchBeta[layerIndex], t));
		}

		return r;

	}

	public float[][] batchNormBackprop(int layerIndex, float[][] dout) {

		int N = dout.length;
		float[] dgamma = Matrix.colsum(Matrix.multiplication(dout, batchLayerNorm[batchIndex][layerIndex]));
		float[] dbeta = Matrix.colsum(dout);

		float[] dh1 = Vector.multiplication(batchGamma[layerIndex], batchSqrtInvVariance[batchIndex][layerIndex],
				1f / N);
		float[][] q = Matrix.multiplication(dout, batchXFixedMean[batchIndex][layerIndex]);
		float[][] dh2 = Matrix.substraction(Matrix.multiply(dout, N),
				Matrix.multiplication(batchXFixedMean[batchIndex][layerIndex],
						Vector.square(batchSqrtInvVariance[batchIndex][layerIndex]), Matrix.colsum(q)),
				dbeta);

		batchGamma[layerIndex] = Vector.substraction(batchGamma[layerIndex],
				Vector.multiply(dgamma, (float) GAMMA_LEARNING_RATE));
		batchBeta[layerIndex] = Vector.substraction(batchBeta[layerIndex],
				Vector.multiply(dbeta, (float) BETA_LEARNING_RATE));
		
		
		return Matrix.multiplication(dh2, dh1);
	}

	public void propagation() {
	
		
		hiddenLayers[0] = activationFunction( batchNormProp(0, Matrix.multiply(inputLayer, synapses[0])));

		for (int i = 1; i < HIDDEN_LAYERS_NUMBER; i++) {

			hiddenLayers[i] = activationFunction( batchNormProp(i, Matrix.multiply(hiddenLayers[i - 1], synapses[i])));

		}
		resultLayer = resultActivationFunction(
				Matrix.multiply(hiddenLayers[HIDDEN_LAYERS_NUMBER - 1], synapses[HIDDEN_LAYERS_NUMBER]));

	}

	public void backpropagation() {
		float[][][] delta = new float[HIDDEN_LAYERS_NUMBER + 1][][];

		float[][][] loss = new float[HIDDEN_LAYERS_NUMBER + 1][][];

		// calculate error
		loss[HIDDEN_LAYERS_NUMBER] = lossFunction(resultLayer, outputLayer);
		
		float[][] grad = resultBackpropagationFunction(resultLayer);

		delta[HIDDEN_LAYERS_NUMBER] = Matrix.multiplication(loss[HIDDEN_LAYERS_NUMBER], grad);
		
		for (int i = HIDDEN_LAYERS_NUMBER - 1; i >= 0; i--) {
			loss[i] = Matrix.multiply(delta[i + 1], Matrix.transpose(synapses[i + 1]));
			
			// get gradient of loss according to normal layer
			delta[i] = Matrix.multiplication(loss[i], backpropagationFunction(hiddenLayers[i]));

			// adjust gradient according to batchnormLayer
			delta[i] = batchNormBackprop(i, delta[i]);// );
			
		}
		float[][] computedSGD, computedRMS;
		float[][] dw = Matrix.multiply(Matrix.transpose(inputLayer), delta[0]);
		
		// update momentum
		weightedAverageUpdate(sgdRunningGradient[0], 0.9f, dw);
		computedSGD = Matrix.divide(sgdRunningGradient[0], (1d - Math.pow(0.9, iteration+1)));
		
		weightedAverageUpdate(rmsRunningGradient[0], 0.999f, Matrix.square(dw));
		computedRMS = Matrix.divide(rmsRunningGradient[0], (1d - Math.pow(0.999, iteration+1)));
		
		
		float[][] correctedRunningGradient = Matrix.multiplication(computedSGD, Matrix.invsqrt(computedRMS, EPSILON));
		synapses[0] = Matrix.substraction(synapses[0],  Matrix.multiply(correctedRunningGradient, (float)LEARNING_RATE));
		
		for (int i = 1; i < HIDDEN_LAYERS_NUMBER + 1 ; i++) {
			dw = Matrix.multiply(Matrix.transpose(hiddenLayers[i - 1]), delta[i]);
			
			
			weightedAverageUpdate(sgdRunningGradient[i], 0.9f, dw);
			//Matrix.print(sgdRunningGradient[i]);
			computedSGD = Matrix.divide(sgdRunningGradient[i], 1d - Math.pow(0.9, iteration+1));

		
			weightedAverageUpdate(rmsRunningGradient[i], 0.999f, Matrix.square(dw));
			computedRMS = Matrix.divide(rmsRunningGradient[i], 1d - Math.pow(0.999, iteration+1));

			
			
			correctedRunningGradient = Matrix.multiplication(computedSGD, Matrix.invsqrt(computedRMS, EPSILON));
			synapses[i] = Matrix.substraction(synapses[i], Matrix.multiply(correctedRunningGradient, (float)LEARNING_RATE));
			
		}
	}

	public float calculateMSE(int bl) {
		float sum = 0;

		for (int i = 0; i < bl; i++)
			sum += Matrix.mean(Matrix.combine(outputBatch[i], resultBatch[i], (m1, m2) -> {float x = m1 - m2; return x*x;}));

		return sum / bl / 2;
	}

	public float getMSE() {
		return mse;
	}

	private void weightedAverageUpdate(float[][] matrix, float beta, float[][] point) {
		int h = matrix.length, w = matrix[0].length;
		//System.out.println(h + " " + point.length + " " + w + " " + point[0].length);
		float alpha = 1f - beta;
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				matrix[j][i] = beta * matrix[j][i] + alpha * point[j][i];
	}

	public void train() {
		train(false);
	}

	public void train(int it) {
		if (inputLayer == null || outputLayer == null)
			throw new IllegalArgumentException("Must first define inputLayer and outputLayer!");

		iteration = it;
		training = true;

		int bl = inputBatch.length;
		resultBatch = new float[bl][][];
		
		// temporary batch norm parameters
		batchLayerNorm = new float[inputBatch.length][HIDDEN_LAYERS_NUMBER][][];
		batchXFixedMean = new float[inputBatch.length][HIDDEN_LAYERS_NUMBER][][];
		batchSqrtInvVariance = new float[inputBatch.length][HIDDEN_LAYERS_NUMBER][];

		//resetRunningGradient();
		
		for (batchIndex = 0; batchIndex < bl; batchIndex++) {
			inputLayer = inputBatch[batchIndex];
			outputLayer = outputBatch[batchIndex];

			propagation();
			backpropagation();

			resultBatch[batchIndex] = resultLayer;

		}

		training = false;
		mse = calculateMSE(bl);

	}

	public void train(boolean log) {
		long avgms = 0;
		long t = System.currentTimeMillis();
		
		if (log)
			System.out.println("Training...");

		for (int n = 0; n < TRAINING_ITERATIONS; n++) {
			train(n);

			if (log) {
				if (n % (TRAINING_ITERATIONS / 10) == 1) {
					long temp = System.currentTimeMillis() - t;
					avgms += temp;
					System.out.println("[MS" + (temp) + "]MSE:" + (mse = calculateMSE(inputBatch.length)));
					t = System.currentTimeMillis();
				}
			}
		}

		if (log)
			System.out.println("Done. [AVGMS" + avgms / 10 + "]");

	}
}
