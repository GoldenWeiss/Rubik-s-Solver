package solver;

import java.util.Arrays;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;

import math.Matrix;
import math.Tensor3d;
import math.Tensor4d;
import math.Vector;

public class ConvNet
{

	private static boolean USE_POOLING = false;
	private static int CONV_LAYERS_NUMBER = 2;
	private static int HIDDEN_CONNECTED_LAYERS_NUMBER = 1;
	private static float BETA = 0.999f;

	private static float BN_MOMENTUM = 0.999f;
	private static float BN_EPSILON = 1e-8f;
	private static float BN_GAMMA_LEARNING_RATE = 1e-2f;
	private static float BN_BETA_LEARNING_RATE = 1e-2f;

	/**
	 * Number of synapses : HIDDEN_CONNECTED_LAYERS_NUMBER + 1
	 */
	private static float[] SYNAPSES_LEARNING_RATES = new float[]
	{ 1e-2f, 1e-2f };

	/**
	 * Number of filters : CONV_LAYERS_NUMBER
	 */
	private static float[] FILTERS_LEARNING_RATES = new float[]
	{ 1e-2f, 1e-2f };

	/**
	 * Number of filters to be applied to each conv layers
	 */
	private static int[] FILTERS_N = new int[]
	{ 32, 32 };

	/**
	 * Size of filters for each conv layer
	 */
	private static int[][] FILTER_SIZE = new int[][]
	{
			{ 3, 3 } , {3, 3}};

	private static int[][] FILTER_STRIDE = new int[][]
	{
			{ 1, 1 } , {1, 1}};

	/**
	 * Size of pooling layers
	 */
	private static int[] POOLING_SIZE = new int[]
	{ 2, 2 };

	// indicate all the fully connected layers size after first flattening
	private static int[] CONNECTED_SIZE = new int[]
	{ 16, 1 };

	private final float[][][][][] filters;
	private float[][][][][] filtersMomentum;

	private final float[][][][][] convLayers;
	private float[][][][][] poolLayers;
	private int[][][][][][] switchLayers;

	private float[][][] connectedLayers;
	private float[][][] connectedSynapses;
	private float[][][] connectedSynapsesMomentum;

	private float[][] outputLayer;

	private int[][] filterLayersSize;
	private int[][] poolLayersSize;

	private int firstConnectedLayerSize;
	private int batchN;
	private int batchSize;

	private int batchIndex;
	private float[][][][][] inputBatch;
	private float[][][] outputBatch;
	private float[][][] resultBatch;

	private int inputChannelNumber;

	private int iteration;
	private boolean training;
	private float mse;

	private float[][][][] bncRunningMean;
	private float[][][][] bncRunningVariance;
	private float[][][][][][] bncXFixedMean;
	private float[][][][][] bncSqrtInvVariance;
	private float[][][][][][] bncLayerNorm;
	private float[][][][] bncBeta, bncGamma;

	private float[][] bnpRunningMean;
	private float[][] bnpRunningVariance;
	private float[][][][] bnpXFixedMean;
	private float[][][] bnpSqrtInvVariance;
	private float[][][][] bnpLayerNorm;
	private float[][] bnpGamma, bnpBeta;

	/**
	 * 
	 * @param pLayerHeight
	 * @param pLayerWidth
	 * @param pLayerDepth number of channels
	 */
	public ConvNet(int pLayerDepth, int pLayerHeight, int pLayerWidth)
	{

		convLayers = new float[CONV_LAYERS_NUMBER + 1][][][][];
		filterLayersSize = new int[CONV_LAYERS_NUMBER + 1][2];
		poolLayersSize = new int[CONV_LAYERS_NUMBER + 1][2];
		filterLayersSize[0][0] = poolLayersSize[0][0] = pLayerHeight;
		filterLayersSize[0][1] = poolLayersSize[0][1] = pLayerWidth;
		inputChannelNumber = pLayerDepth;

		filters = new float[CONV_LAYERS_NUMBER][][][][];
		filtersMomentum = new float[CONV_LAYERS_NUMBER][][][][];

		for (int n = 0; n < CONV_LAYERS_NUMBER; n++)
		{
			filterLayersSize[n + 1][0] = poolLayersSize[n
					+ 1][0] = (poolLayersSize[n][0] - FILTER_SIZE[n][0])
							/ FILTER_STRIDE[n][0] + 1;
			filterLayersSize[n + 1][1] = poolLayersSize[n
					+ 1][1] = (poolLayersSize[n][1] - FILTER_SIZE[n][1])
							/ FILTER_STRIDE[n][1] + 1;

			if (USE_POOLING)
			{
				poolLayersSize[n + 1][0] = filterLayersSize[n + 1][0] / 2;
				poolLayersSize[n + 1][1] = filterLayersSize[n + 1][1] / 2;
			}

			filters[n] = new float[FILTERS_N[n]][][][];
			filtersMomentum[n] = new float[FILTERS_N[n]][][][];
			for (int o = 0; o < FILTERS_N[n]; o++)
			{
				int depth = n == 0 ? inputChannelNumber : FILTERS_N[n - 1];
				filters[n][o] = Tensor3d.build(depth, FILTER_SIZE[n][0],
						FILTER_SIZE[n][1], (k, j, i) -> weightRandomValue());
				filtersMomentum[n][o] = new float[depth][FILTER_SIZE[n][0]][FILTER_SIZE[n][1]];
			}

		}

		firstConnectedLayerSize = filterLayersSize[CONV_LAYERS_NUMBER][0]
				* filterLayersSize[CONV_LAYERS_NUMBER][1]
				* FILTERS_N[CONV_LAYERS_NUMBER - 1];

		// System.out.println(filterLayersSize[0][0]);
		// convLayer[0] = inputLayer

		/*
		 * connectedLayers[0] = flatten(convLayers[CONV_LAYERS_NUMBER])
		 * connectedLayers[HIDDEN_CONNECTED_LAYERS_NUMBER+1] = resultLayer
		 */

		connectedSynapses = new float[HIDDEN_CONNECTED_LAYERS_NUMBER + 1][][];
		connectedSynapsesMomentum = new float[HIDDEN_CONNECTED_LAYERS_NUMBER
				+ 1][][];

		connectedSynapses[0] = Matrix.build(firstConnectedLayerSize,
				CONNECTED_SIZE[0], (i, j) -> weightRandomValue());
		connectedSynapsesMomentum[0] = new float[firstConnectedLayerSize][CONNECTED_SIZE[0]];

		if (HIDDEN_CONNECTED_LAYERS_NUMBER > 0)
			for (int n = 1; n < HIDDEN_CONNECTED_LAYERS_NUMBER + 1; n++)
			{
				connectedSynapses[n] = Matrix.build(CONNECTED_SIZE[n - 1],
						CONNECTED_SIZE[n], (i, j) -> weightRandomValue());
				connectedSynapsesMomentum[n] = new float[CONNECTED_SIZE[n
						- 1]][CONNECTED_SIZE[n]];
			}

		initBN();
		// resetMomentums();
	}

	private float weightRandomValue()
	{
		return (float) (2 * Math.random() - 1) * 1e-2f;
	}

	public void setBatchSize(int pBatchSize)
	{
		if (pBatchSize > 0)
			batchSize = pBatchSize;

		for (int i = 0; i < CONV_LAYERS_NUMBER + 1; i++)
			convLayers[i] = new float[batchSize][][][];
		poolLayers = new float[CONV_LAYERS_NUMBER + 1][batchSize][][][];
		switchLayers = new int[CONV_LAYERS_NUMBER + 1][batchSize][][][][];
		connectedLayers = new float[HIDDEN_CONNECTED_LAYERS_NUMBER
				+ 2][batchSize][];
		connectedLayers[0] = new float[batchSize][firstConnectedLayerSize];

		bncLayerNorm = new float[batchSize][CONV_LAYERS_NUMBER][][][][];
		bncXFixedMean = new float[batchSize][CONV_LAYERS_NUMBER][][][][];
		bncSqrtInvVariance = new float[batchSize][CONV_LAYERS_NUMBER][][][];

		bnpLayerNorm = new float[batchSize][HIDDEN_CONNECTED_LAYERS_NUMBER][][];
		bnpXFixedMean = new float[batchSize][HIDDEN_CONNECTED_LAYERS_NUMBER][][];
		bnpSqrtInvVariance = new float[batchSize][HIDDEN_CONNECTED_LAYERS_NUMBER][];
	}

	public void setInputLayer(float[][][][] pInputLayer)
	{
		if (batchSize == 0)
			throw new IllegalArgumentException("Must first define batchSize!");

		if (inputChannelNumber != pInputLayer[0].length
				|| pInputLayer[0][0].length != filterLayersSize[0][0]
				|| pInputLayer[0][0][0].length != filterLayersSize[0][1])
			throw new IllegalArgumentException("Format error! pInputLayer[0] :"
					+ pInputLayer[0].length + ";inputChannelNumber :"
					+ inputChannelNumber + ";pInputLayer[0][0] :"

					+ pInputLayer[0][0].length + ";filterLayersSize[0][0] :"
					+ filterLayersSize[0][0] + ";pInputLayer[0][0][0] :"
					+ pInputLayer[0][0][0].length + ";filterLayersSize[0][1] :"
					+ filterLayersSize[0][1]);

		inputBatch = Matrix.toBatch(batchSize, pInputLayer);
		batchN = inputBatch.length;

		convLayers[0] = poolLayers[0] = inputBatch[0];

		resultBatch = new float[batchN][][];

	}

	public void setOutputLayer(float[][] pOutputLayer)
	{
		outputBatch = Matrix.toBatch(batchSize, pOutputLayer);
		outputLayer = outputBatch[0];

	}

	public float[][] getResultLayer()
	{
		return connectedLayers[HIDDEN_CONNECTED_LAYERS_NUMBER + 1];
	}

	private float leakyRelu(float x)
	{
		return x >= 0 ? x : 0.01f * x;
	}

	private float leakyReluInverse(float x)
	{
		return x >= 0 ? 1f : 0.01f;
	}

	private void applyPooling(int pFilterIndex, float[][][] pConv,
			int[][][][] pSwitch, float[][][] pPool)
	{

		int pSize = POOLING_SIZE[pFilterIndex - 1];
		int yReal, xReal;

		for (int k = 0; k < FILTERS_N[pFilterIndex - 1]; k++)
			for (int j = 0; j < poolLayersSize[pFilterIndex][0]; j++)
				for (int i = 0; i < poolLayersSize[pFilterIndex][1]; i++)
				{
					pPool[k][j][i] = Float.NEGATIVE_INFINITY; // si on a
																// explosion du
																// gradient :d
					for (int y = 0; y < pSize; y++)
						for (int x = 0; x < pSize; x++)
							if ((pConv[k][yReal = j * pSize + y][xReal = i
									* pSize + x]) >= pPool[k][j][i])
							{
								pPool[k][j][i] = pConv[k][yReal][xReal];
								pSwitch[k][j][i][0] = yReal;
								pSwitch[k][j][i][1] = xReal;
							}
				}
	}

	private void flattenLayer(int pConvIndex)
	{
		float[][][][] pFeaturesMaps1;
		int h, w;
		if (USE_POOLING)
		{
			pFeaturesMaps1 = poolLayers[pConvIndex];
			h = poolLayersSize[pConvIndex][0];
			w = poolLayersSize[pConvIndex][1];
		}
		else
		{
			pFeaturesMaps1 = convLayers[pConvIndex];
			h = filterLayersSize[pConvIndex][0];
			w = filterLayersSize[pConvIndex][1];
		}
		for (int b = 0; b < batchSize; b++)
			for (int n = 0; n < FILTERS_N[pConvIndex - 1]; n++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
					{
						// System.out.println(connectedLayers[0][0]);
						connectedLayers[0][b][n * h * w + j * w
								+ i] = pFeaturesMaps1[b][n][j][i];
					}
		// System.out.println((FILTERS_N[pConvIndex - 1]) * h * w + " " +
		// firstConnectedLayerSize);
	}

	private float[][][][] unflattenLayer(int pConvIndex,
			float[][] connectedLayer)
	{
		int h, w;
		if (USE_POOLING)
		{
			h = poolLayersSize[pConvIndex][0];
			w = poolLayersSize[pConvIndex][1];
		}
		else
		{
			h = filterLayersSize[pConvIndex][0];
			w = filterLayersSize[pConvIndex][1];
		}

		float[][][][] rFeaturesMaps = new float[batchSize][FILTERS_N[pConvIndex
				- 1]][h][w];
		for (int b = 0; b < batchSize; b++)
			for (int n = 0; n < FILTERS_N[pConvIndex - 1]; n++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						rFeaturesMaps[b][n][j][i] = connectedLayer[b][n * h * w
								+ j * w + i];

		return rFeaturesMaps;

	}

	/**
	 * Convolution operation [padding: 0; stride: 1]
	 * 
	 * @return
	 */
	private void applyConvolution(int pFilterIndex, float[][][] pFeaturesMaps,
			float[][][] pConv)
	{

		int h = filterLayersSize[pFilterIndex][0];
		int w = filterLayersSize[pFilterIndex][1];

		int inputMapsNumber = (pFilterIndex == 1 ? inputChannelNumber
				: FILTERS_N[pFilterIndex - 2]);

		int saved = pFilterIndex - 1;

		for (int k = 0; k < FILTERS_N[pFilterIndex - 1]; k++)
		{
			for (int n = 0; n < inputMapsNumber; n++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						for (int y = 0; y < FILTER_SIZE[saved][0]; y++)
							for (int x = 0; x < FILTER_SIZE[saved][1]; x++)
							{ //
								
								pConv[k][j][i] += pFeaturesMaps[n][j
										* FILTER_STRIDE[saved][0] + y][i
												* FILTER_STRIDE[saved][1] + x]
										* filters[pFilterIndex - 1][k][n][y][x];
							}
		}

	}

	private float[][][] applyActivation(int pFilterIndex, float[][][] pConv)
	{

		for (int k = 0; k < FILTERS_N[pFilterIndex - 1]; k++)
			for (int j = 0; j < filterLayersSize[pFilterIndex][0]; j++)
				for (int i = 0; i < filterLayersSize[pFilterIndex][1]; i++)
					pConv[k][j][i] = leakyRelu(pConv[k][j][i]);

		return pConv;

	}

	// TODO change BN with args convLayers[n] not convLayers[n][i]
	public void propagation()
	{
		for (int n = 0; n < CONV_LAYERS_NUMBER; n++)
		{
			for (int i = 0; i < batchSize; i++)
			{
				convLayers[n
						+ 1][i] = new float[FILTERS_N[n]][filterLayersSize[n
								+ 1][0]][filterLayersSize[n + 1][1]];
				poolLayers[n + 1][i] = new float[FILTERS_N[n]][poolLayersSize[n
						+ 1][0]][poolLayersSize[n + 1][1]];
				switchLayers[n + 1][i] = new int[FILTERS_N[n]][poolLayersSize[n
						+ 1][0]][poolLayersSize[n + 1][1]][2];

				// System.out.println(n);
				applyConvolution(n + 1, convLayers[n][i], convLayers[n + 1][i]);
			}
			convLayers[n + 1] = applyBN(n, convLayers[n + 1]);
		}
		for (int n = 0; n < CONV_LAYERS_NUMBER; n++)
			for (int i = 0; i < batchSize; i++)
			{
				applyActivation(n + 1, convLayers[n + 1][i]);

				if (USE_POOLING)
					applyPooling(n + 1, convLayers[n + 1][i],
							switchLayers[n + 1][i], poolLayers[n + 1][i]);
			}

		flattenLayer(CONV_LAYERS_NUMBER);

		for (int i = 1; i < HIDDEN_CONNECTED_LAYERS_NUMBER + 2; i++)
		{
			connectedLayers[i] = Matrix.multiply(connectedLayers[i - 1],
					connectedSynapses[i - 1]);
			if (i < HIDDEN_CONNECTED_LAYERS_NUMBER + 1)
				connectedLayers[i] = applyBN(i - 1, connectedLayers[i]);
			connectedLayers[i] = Matrix.apply(connectedLayers[i],
					e -> leakyRelu(e));
		}
		resultBatch[batchIndex] = getResultLayer();
	}

	private void weightedAverageUpdate(float[] vector, float beta,
			float[] point)
	{
		float alpha = 1f - beta;
		for (int i = 0, w = vector.length; i < w; i++)
			vector[i] = beta * vector[i] + alpha * vector[i];
	}

	private void weightedAverageUpdate(float[][] matrix, float beta,
			float[][] point)
	{
		float alpha = 1f - beta;
		for (int j = 0, h = matrix.length; j < h; j++)
			for (int i = 0, w = matrix[0].length; i < w; i++)
				matrix[j][i] = beta * matrix[j][i] + alpha * point[j][i];
	}

	private void weightedAverageUpdate(float[][][] tensor, float beta,
			float[][][] point)
	{
		float alpha = 1f - beta;
		for (int k = 0, d = tensor.length; k < d; k++)
			for (int j = 0, h = tensor[0].length; j < h; j++)
				for (int i = 0, w = tensor[0][0].length; i < w; i++)
					tensor[k][j][i] = beta * tensor[k][j][i]
							+ alpha * point[k][j][i];

	}

	// TODO
	private void backpropagation()
	{

		// 1
		float[][][] deltaConnected = new float[HIDDEN_CONNECTED_LAYERS_NUMBER
				+ 2][][];
		float[][][] lossConnected = new float[HIDDEN_CONNECTED_LAYERS_NUMBER
				+ 2][][];

		lossConnected[HIDDEN_CONNECTED_LAYERS_NUMBER + 1] = Matrix
				.substraction(getResultLayer(), outputLayer);
		for (int i = HIDDEN_CONNECTED_LAYERS_NUMBER + 1; i > 0; i--)
		{
			deltaConnected[i] = Matrix.multiplication(lossConnected[i],
					Matrix.apply(connectedLayers[i], e -> leakyReluInverse(e)));
			if (i < HIDDEN_CONNECTED_LAYERS_NUMBER + 1)
				deltaConnected[i] = getDwBN(i - 1, deltaConnected[i]);
			// System.out.println(connectedSynapses[i]);
			lossConnected[i - 1] = Matrix.multiply(deltaConnected[i],
					Matrix.transpose(connectedSynapses[i - 1]));
		}
		deltaConnected[0] = Matrix.multiplication(lossConnected[0],
				Matrix.apply(connectedLayers[0], e -> leakyReluInverse(e)));

		// TODO sgd without momentum or adam for the moment
		for (int i = 0; i < HIDDEN_CONNECTED_LAYERS_NUMBER + 1; i++)
		{
			weightedAverageUpdate(connectedSynapsesMomentum[i], BETA,
					Matrix.multiply(Matrix.transpose(connectedLayers[i]),
							deltaConnected[i + 1]));
			Matrix.substract(connectedSynapses[i], connectedSynapsesMomentum[i],
					SYNAPSES_LEARNING_RATES[i]);
		}

		// 2

		float[][][][][] deltaConv = new float[CONV_LAYERS_NUMBER][][][][];
		float[][][][][] lossConv = new float[CONV_LAYERS_NUMBER][][][][];
		float[][][][][] dwConv = new float[CONV_LAYERS_NUMBER][][][][];

		float[][][][] a = unflattenLayer(CONV_LAYERS_NUMBER, deltaConnected[0]);

		a = getDwBN(CONV_LAYERS_NUMBER - 1, a);

		deltaConv[CONV_LAYERS_NUMBER - 1] = poolingBackprop(
				CONV_LAYERS_NUMBER - 1, a, switchLayers[CONV_LAYERS_NUMBER]);
		dwConv[CONV_LAYERS_NUMBER - 1] = getDwConv(CONV_LAYERS_NUMBER - 1,
				deltaConv[CONV_LAYERS_NUMBER - 1],
				convLayers[CONV_LAYERS_NUMBER - 1]);

		if (CONV_LAYERS_NUMBER > 1)
		{

			lossConv[CONV_LAYERS_NUMBER - 2] = propagateLossConv(
					CONV_LAYERS_NUMBER - 1, deltaConv[CONV_LAYERS_NUMBER - 1]);

			for (int i = CONV_LAYERS_NUMBER - 2; i >= 0; i--)
			{
				deltaConv[i] = getDeltaConv(i + 1, lossConv[i],
						convLayers[i + 1]);

				deltaConv[i] = getDwBN(i, deltaConv[i]);

				deltaConv[i] = poolingBackprop(i, deltaConv[i],
						switchLayers[i]);

				dwConv[i] = getDwConv(i, deltaConv[i], convLayers[i]);

				if (i > 0)
					lossConv[i - 1] = propagateLossConv(i, deltaConv[i]);
			}
			// } else {

		}

		for (int i = 0; i < CONV_LAYERS_NUMBER; i++)
			for (int n = 0; n < FILTERS_N[i]; n++)
			{

				weightedAverageUpdate(filtersMomentum[i][n], BETA,
						dwConv[i][n]);
				Tensor3d.substract(filters[i][n], filtersMomentum[i][n],
						FILTERS_LEARNING_RATES[i]);

			}

	}

	private float[][][][] poolingBackprop(int pFilterIndex, float[][][][] delta,
			int[][][][][] switches)
	{
		if (USE_POOLING)
		{
			float[][][][] pConv = new float[batchSize][FILTERS_N[pFilterIndex
					- 1]][filterLayersSize[pFilterIndex][0]][filterLayersSize[pFilterIndex][1]];
			for (int b = 0; b < batchSize; b++)
				for (int k = 0; k < FILTERS_N[pFilterIndex - 1]; k++)
					for (int j = 0; j < poolLayersSize[pFilterIndex][0]; j++)
						for (int i = 0; i < poolLayersSize[pFilterIndex][1]; i++)
							pConv[b][k][switches[b][k][j][i][0]][switches[b][k][j][i][1]] = delta[b][k][j][i];
			return pConv;
		}
		else
		{
			return delta;
		}
	}

	private float[][][][] getDwConv(int pFilterIndex, float[][][][] delta,
			float[][][][] previousConv)
	{
		int saved = pFilterIndex + 1;
		int h = filterLayersSize[saved][0], w = filterLayersSize[saved][1];

		int inputMapsNumber = (pFilterIndex == 0 ? inputChannelNumber
				: FILTERS_N[pFilterIndex - 1]);
		float[][][][] dw = new float[FILTERS_N[pFilterIndex]][inputMapsNumber][FILTER_SIZE[pFilterIndex][0]][FILTER_SIZE[pFilterIndex][1]];
		/* E/dw = E/dx * dx/dw (dx/dw varies from input before convolution) */
		for (int b = 0; b < batchSize; b++)
		{
			for (int k = 0; k < FILTERS_N[pFilterIndex]; k++)
				for (int n = 0; n < inputMapsNumber; n++)
				{

					for (int j = 0; j < h; j++)
						for (int i = 0; i < w; i++)
							for (int y = 0; y < FILTER_SIZE[pFilterIndex][0]; y++)
								for (int x = 0; x < FILTER_SIZE[pFilterIndex][1]; x++)
								{
									/*
									 * gradients are summed over the training
									 * (batch) samples toflip(previousConv):
									 * simply reverse [j+y] with [i+x]
									 */
									// System.out.println(previousConv[b][n].length);

									dw[k][n][y][x] += previousConv[b][n][j
											* FILTER_STRIDE[pFilterIndex][0]
											+ y][(i) * FILTER_STRIDE[pFilterIndex][1]
													+ x]
											* delta[b][k][j][i];

								}

				}
		}
		return dw;
	}

	private float[][][][] propagateLossConv(int pFilterIndex,
			float[][][][] delta)
	{
		int saved = pFilterIndex + 1;
		int h = filterLayersSize[saved][0], w = filterLayersSize[saved][1];

		int inputMapsNumber = (pFilterIndex == 0 ? inputChannelNumber
				: FILTERS_N[pFilterIndex - 1]);

		float[][][][] ploss = new float[batchSize][inputMapsNumber][filterLayersSize[pFilterIndex][0]][filterLayersSize[pFilterIndex][1]];
		// System.out.println(filterLayersSize[pFilterIndex][0] + " " + (h +
		// FILTER_SIZE[pFilterIndex]));
		for (int b = 0; b < batchSize; b++)
			for (int n = 0; n < FILTERS_N[pFilterIndex]; n++)
				for (int j = 0; j < filterLayersSize[pFilterIndex][0]; j++)
					for (int i = 0; i < filterLayersSize[pFilterIndex][1]; i++)
						for (int k = 0; k < inputMapsNumber; k++)
							for (int y = 0; y < FILTER_SIZE[pFilterIndex][0]; y++)
								for (int x = 0; x < FILTER_SIZE[pFilterIndex][1]; x++)
								{
									/*
									 * propagate loss dE/dy(l-1) = de/dx *
									 * dx/dy(l-1) (dx/dy(l-1) = the previous
									 * filter
									 */

									// TODO Flip filters? THE MOST IMPORTANT
									// LINE OF CODEE HERE IS TO FLIP!!!!!
									int ty = j + y, tx = i + x;

									// } else
									if (ty >= FILTER_SIZE[pFilterIndex][0] - 1
											&& (ty - FILTER_SIZE[pFilterIndex][0]
													+ 1)
													% (FILTER_STRIDE[pFilterIndex][0]) == 0
											&& tx >= FILTER_SIZE[pFilterIndex][1]
													- 1
											&& (tx - FILTER_SIZE[pFilterIndex][1]
													+ 1)
													% (FILTER_STRIDE[pFilterIndex][1]) == 0
											&& ty < filterLayersSize[pFilterIndex][0]
													- FILTER_SIZE[pFilterIndex][0]
													+ 1
											&& tx < filterLayersSize[pFilterIndex][1]
													- FILTER_SIZE[pFilterIndex][1]
													+ 1)
									{

										// TODO print true
										ploss[b][k][j][i] += delta[b][n][(j + y
												- FILTER_SIZE[pFilterIndex][0]
												+ 1)
												/ FILTER_STRIDE[pFilterIndex][0]][(i
														+ x
														- FILTER_SIZE[pFilterIndex][1]
														+ 1)
														/ FILTER_STRIDE[pFilterIndex][1]]
												* filters[pFilterIndex][n][k][FILTER_SIZE[pFilterIndex][0]
														- y
														- 1][FILTER_SIZE[pFilterIndex][1]
																- x - 1];
									}
								}

		return ploss;
	}

	private float[][][][] getDeltaConv(int pFilterIndex, float[][][][] ploss,
			float[][][][] pPool)
	{
		int h = filterLayersSize[pFilterIndex][0];
		int w = filterLayersSize[pFilterIndex][1];

		int inputMapsNumber = FILTERS_N[pFilterIndex - 1];// (pFilterIndex == 0
															// ?
															// inputChannelNumber
															// :
															// FILTERS_N[pFilterIndex
															// - 1]);

		float[][][][] delta = new float[batchSize][inputMapsNumber][h][w];
		for (int b = 0; b < batchSize; b++)
			for (int k = 0; k < inputMapsNumber; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
					{
						delta[b][k][j][i] = ploss[b][k][j][i]
								* leakyReluInverse(pPool[b][k][j][i]);
					}

		return delta;
	}

	private void initBN()
	{
		bncGamma = new float[CONV_LAYERS_NUMBER][][][];
		bncBeta = new float[CONV_LAYERS_NUMBER][][][];
		bncRunningMean = new float[CONV_LAYERS_NUMBER][][][];
		bncRunningVariance = new float[CONV_LAYERS_NUMBER][][][];

		bnpGamma = new float[HIDDEN_CONNECTED_LAYERS_NUMBER][];
		bnpBeta = new float[HIDDEN_CONNECTED_LAYERS_NUMBER][];
		bnpRunningMean = new float[HIDDEN_CONNECTED_LAYERS_NUMBER][];
		bnpRunningVariance = new float[HIDDEN_CONNECTED_LAYERS_NUMBER][];

		for (int n = 0; n < CONV_LAYERS_NUMBER; n++)
		{
			int h = filterLayersSize[n + 1][0], w = filterLayersSize[n + 1][1];
			bncGamma[n] = new float[FILTERS_N[n]][h][w];
			bncBeta[n] = new float[FILTERS_N[n]][h][w];
			bncRunningMean[n] = new float[FILTERS_N[n]][h][w];
			bncRunningVariance[n] = new float[FILTERS_N[n]][h][w];

			for (int f = 0; f < FILTERS_N[n]; f++)
			{
				for (int j = 0; j < h; j++)
				{
					for (int i = 0; i < w; i++)
					{
						bncGamma[n][f][j][i] = 1;
						bncBeta[n][f][j][i] = 0;
						bncRunningMean[n][f][j][i] = 0;
						bncRunningVariance[n][f][j][i] = 1;
					}
				}
			}
		}

		for (int n = 0; n < HIDDEN_CONNECTED_LAYERS_NUMBER; n++)
		{
			bnpGamma[n] = new float[CONNECTED_SIZE[n]];
			bnpBeta[n] = new float[CONNECTED_SIZE[n]];
			bnpRunningMean[n] = new float[CONNECTED_SIZE[n]];
			bnpRunningVariance[n] = new float[CONNECTED_SIZE[n]];
			Arrays.fill(bnpGamma[n], 1);
			Arrays.fill(bnpBeta[n], 0);
			Arrays.fill(bnpRunningMean[n], 0);
			Arrays.fill(bnpRunningVariance[n], 1);
		}

	}

	public float[][][][] applyBN(int pLayerIndex, float[][][][] pInput)
	{
		float[][][][] r = null;
		float[][][] mean, variance;

		if (training)
		{

			mean = Tensor4d.tMean(pInput);
			bncXFixedMean[batchIndex][pLayerIndex] = Tensor4d
					.substraction(pInput, mean);
			variance = Tensor4d.tMean(
					Tensor4d.square(bncXFixedMean[batchIndex][pLayerIndex]));

			weightedAverageUpdate(bncRunningMean[pLayerIndex], BN_MOMENTUM,
					mean);
			weightedAverageUpdate(bncRunningVariance[pLayerIndex], BN_MOMENTUM,
					variance);

			bncSqrtInvVariance[batchIndex][pLayerIndex] = Tensor3d
					.invsqrt(variance, BN_EPSILON);
			bncLayerNorm[batchIndex][pLayerIndex] = Tensor4d.multiplication(
					bncXFixedMean[batchIndex][pLayerIndex],
					bncSqrtInvVariance[batchIndex][pLayerIndex]);

			r = Tensor4d.multiplication(bncLayerNorm[batchIndex][pLayerIndex],
					bncGamma[pLayerIndex]);
			Tensor4d.add(r, bncBeta[pLayerIndex]);
		}
		else
		{
			mean = bncRunningMean[pLayerIndex];
			variance = bncRunningVariance[pLayerIndex];
			// variance = Tensor3d.multiply(bncRunningVariance[pLayerIndex],
			// batchSize / Math.max(batchSize - 1f, 1));

			Tensor4d.substraction_(pInput, mean);
			Tensor4d.multiplication_(pInput,
					Tensor3d.invsqrt(variance, BN_EPSILON),
					bncGamma[pLayerIndex]);
			Tensor4d.add(pInput, bncBeta[pLayerIndex]);

			/*
			 * float[][][] d = Tensor3d.multiplication(bncGamma[pLayerIndex],
			 * Tensor3d.invsqrt(variance, BN_EPSILON)); float[][][] t =
			 * Tensor3d.multiplication(d, mean);
			 * Tensor4d.multiplication_(pInput, d); Tensor4d.add(pInput,
			 * Tensor3d.substraction(bncBeta[pLayerIndex], t));
			 */
			r = pInput;
		}

		return r;
	}

	public float[][][][] getDwBN(int pLayerIndex, float[][][][] dout)
	{
		int N = dout.length;
		Tensor3d.multiply_(bncSqrtInvVariance[batchIndex][pLayerIndex],  1f / (float)Math.sqrt(N));
		float[][][] dgamma = Tensor4d.tSum(Tensor4d.multiplication(dout,
				bncLayerNorm[batchIndex][pLayerIndex]));
		float[][][] dbeta = Tensor4d.tSum(dout);

		float[][][] dh1 = Tensor3d.multiplication(bncGamma[pLayerIndex],
				bncSqrtInvVariance[batchIndex][pLayerIndex], 1f / N);
		float[][][][] q = Tensor4d.multiplication(dout,
				bncXFixedMean[batchIndex][pLayerIndex]);

		Tensor4d.multiply_(dout, N);
		Tensor3d.square_(bncSqrtInvVariance[batchIndex][pLayerIndex]);
		Tensor4d.multiplication_(bncXFixedMean[batchIndex][pLayerIndex],
				bncSqrtInvVariance[batchIndex][pLayerIndex], Tensor4d.tSum(q));

		Tensor4d.substraction_(dout, bncXFixedMean[batchIndex][pLayerIndex],
				dbeta);

		Tensor3d.substract(bncGamma[pLayerIndex], dgamma,
				BN_GAMMA_LEARNING_RATE);
		Tensor3d.substract(bncBeta[pLayerIndex], dbeta, BN_BETA_LEARNING_RATE);

		Tensor4d.multiplication_(dout, dh1);

		return dout;
	}

	public float[][] applyBN(int pLayerIndex, float[][] pInput)
	{

		float[][] r = null;

		float[] mean;
		float[] variance;

		if (training)
		{
			mean = Matrix.colmean(pInput);
			bnpXFixedMean[batchIndex][pLayerIndex] = Matrix.substraction(pInput,
					mean);
			variance = Matrix.colmean(
					Matrix.square(bnpXFixedMean[batchIndex][pLayerIndex]));

			weightedAverageUpdate(bnpRunningMean[pLayerIndex], BN_MOMENTUM,
					mean);
			weightedAverageUpdate(bnpRunningVariance[pLayerIndex], BN_MOMENTUM,
					variance);

			bnpSqrtInvVariance[batchIndex][pLayerIndex] = Vector
					.sqrtinv(variance, BN_EPSILON);
			bnpLayerNorm[batchIndex][pLayerIndex] = Matrix.multiplication(
					bnpXFixedMean[batchIndex][pLayerIndex],
					bnpSqrtInvVariance[batchIndex][pLayerIndex]);

			r = Matrix.multiplication(bnpLayerNorm[batchIndex][pLayerIndex],
					bnpGamma[pLayerIndex]);
			Matrix.addition_(r, bnpBeta[pLayerIndex]);
		}
		else
		{
			mean = bnpRunningMean[pLayerIndex];
			variance = bnpRunningVariance[pLayerIndex];
			// variance = Vector.multiply(bnpRunningVariance[pLayerIndex],
			// batchSize / Math.max(batchSize - 1f, 1));

			Matrix.substraction_(pInput, mean);
			Matrix.multiplication_(pInput, Vector.sqrtinv(variance, BN_EPSILON),
					bnpGamma[pLayerIndex]);
			Matrix.addition_(pInput, bnpBeta[pLayerIndex]);

			/*
			 * float[] d = Vector.division(bnpGamma[pLayerIndex],
			 * Vector.sqrt(variance, BN_EPSILON)); float[] t =
			 * Vector.multiplication(d, mean); Matrix.multiplication_(pInput,
			 * d); Matrix.addition_(pInput,
			 * Vector.substraction(bnpBeta[pLayerIndex], t));
			 */
			r = pInput;
		}
		return r;
	}

	public float[][] getDwBN(int pLayerIndex, float[][] dout)
	{
		int N = dout.length;
		Vector.multiply_(bnpSqrtInvVariance[batchIndex][pLayerIndex], (float) (1f / Math.sqrt(N)));
		float[] dgamma = Matrix.colsum(Matrix.multiplication(dout,
				bnpLayerNorm[batchIndex][pLayerIndex]));
		float[] dbeta = Matrix.colsum(dout);

		float[] dh1 = Vector.multiplication(bnpGamma[pLayerIndex],
				bnpSqrtInvVariance[batchIndex][pLayerIndex], 1f / N);
		float[][] q = Matrix.multiplication(dout,
				bnpXFixedMean[batchIndex][pLayerIndex]);
		Matrix.multiply_(dout, N);
		Vector.square_(bnpSqrtInvVariance[batchIndex][pLayerIndex]);
		Matrix.multiplication_(bnpXFixedMean[batchIndex][pLayerIndex],
				bnpSqrtInvVariance[batchIndex][pLayerIndex], Matrix.colsum(q));
		Matrix.substraction_(dout, bnpXFixedMean[batchIndex][pLayerIndex],
				dbeta);

		Vector.substract_(bnpGamma[pLayerIndex], dgamma,
				BN_GAMMA_LEARNING_RATE);
		Vector.substract_(bnpBeta[pLayerIndex], dbeta, BN_BETA_LEARNING_RATE);

		Matrix.multiplication_(dout, dh1);

		return dout;
	}

	private void calculateMSE()
	{
		float sum = 0;

		for (int i = 0; i < batchN; i++)
			sum += Matrix.mean(Matrix.combine(outputBatch[i], resultBatch[i],
					(m1, m2) -> (m1 - m2) * (m1 - m2)));

		mse = sum / batchN / 2;
	}

	public float getMSE()
	{
		return mse;
	}

	private void resetMomentums()
	{
		for (int n = 0; n < CONV_LAYERS_NUMBER; n++)
		{
			filtersMomentum[n] = new float[FILTERS_N[n]][][][];
			for (int o = 0; o < FILTERS_N[n]; o++)
			{
				int depth = n == 0 ? inputChannelNumber : FILTERS_N[n - 1];
				filtersMomentum[n][o] = new float[depth][FILTER_SIZE[n][0]][FILTER_SIZE[n][1]];
			}
		}
		connectedSynapsesMomentum = new float[HIDDEN_CONNECTED_LAYERS_NUMBER
				+ 1][][];
		connectedSynapsesMomentum[0] = new float[firstConnectedLayerSize][CONNECTED_SIZE[0]];

		if (HIDDEN_CONNECTED_LAYERS_NUMBER > 0)
			for (int n = 1; n < HIDDEN_CONNECTED_LAYERS_NUMBER + 1; n++)
				connectedSynapsesMomentum[n] = new float[CONNECTED_SIZE[n
						- 1]][CONNECTED_SIZE[n]];
	}

	public void train(int it)
	{
		iteration = it;
		training = true;

		for (batchIndex = 0; batchIndex < batchN; batchIndex++)
		{
			resetMomentums();

			convLayers[0] = poolLayers[0] = inputBatch[batchIndex];
			outputLayer = outputBatch[batchIndex];

			propagation();
			backpropagation();
			// propagation();

		}

		batchIndex = 0;

		calculateMSE();
		training = false;
	}

	public void propagate()
	{
		for (batchIndex = 0; batchIndex < batchN; batchIndex++)
		{

			convLayers[0] = poolLayers[0] = inputBatch[batchIndex];
			propagation();
		}
		batchIndex = 0;
	}

	public float[][][][][] getInputBatch()
	{
		return inputBatch;
	}

	public float[][][] getResultBatch()
	{
		return resultBatch;
	}

	public void setBatchIndex(int pVal)
	{
		batchIndex = pVal;
	}
}
