package solver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.aparapi.internal.kernel.KernelPreferences;

import cube.Cube;
import cube.RNotation;
import cube.SpriteCube;
import math.Matrix;

public class CNeuralInterface implements Runnable
{

	private static final int CHECK_LOSS = 5;
	private static final int TRAINING_ITERATIONS = 30_000;
	private static final int THRESHOLD_ITERATIONS = 40;
	private static final int SAMPLE_SIZE = 50000;
	private static final int BATCH_SIZE = 1000;
	private static final int K = Cube.GOD_NUMBER + 4;
	private static final float LOSS_THRESHOLD = 0.05f;

	private SpriteCube sc;
	private ConvNet cn;
	private List<int[][][][]> record;
	private List<Integer> depth;
	private List<float[][][]> X;
	private List<float[]> y;
	private int[][][][] state;
	private float[][][] channelState;
	private float[][][][] inputMatrix;
	private int t;
	private static int[][][][] goalState;

	public int n;
	private float[][] outputMatrix;

	public CNeuralInterface(SpriteCube psc)
	{
		Matrix.print(Cube.toChannel4(Cube.loadSolved())[0]);
		sc = psc;
		cn = new ConvNet(1, 16, 5); // each face is treated as a channel
		cn.setBatchSize(BATCH_SIZE);

		record = new ArrayList<>();
		depth = new ArrayList<>();
		X = new ArrayList<>();
		y = new ArrayList<>();
		t = -1;
		n = 0;
		goalState = Cube.loadSolved();
	}

	public void run()
	{
		genStateSamples();

		for (n = 0; n < TRAINING_ITERATIONS && t < THRESHOLD_ITERATIONS; n++)
		{

			System.out.println("[n=" + n + "; MSE=" + cn.getMSE() + "]");

			if (n % CHECK_LOSS == 0 && cn.getMSE() <= LOSS_THRESHOLD)
			{
				if (n > 0)
				{
					System.out.println(cn.getResultBatch()[0][0][0]);

					channelState = cn.getInputBatch()[0][0];

					cn.setBatchSize(BATCH_SIZE);
					cn.setInputLayer(inputMatrix);
					cn.propagate();
					System.out.println(cn.getResultBatch()[0][0][0]);
				}

				System.out.println("Reached threshold [E=" + LOSS_THRESHOLD
						+ "; t=" + ++t + "]");
				if (t < THRESHOLD_ITERATIONS)
				{
					System.out
							.print("Diffusing new output to expected output [");
					y.clear();
					System.gc();
					float temp;
					int total = 0;

					for (int i = 0, a = record.size(); i < a; i++)
					{
						int[][][][] s = record.get(i);

						temp = Float.MAX_VALUE;
						for (RNotation m : RNotation.values())
						{

							state = Cube.rotate(s, m);
							channelState = Cube.toChannel(state);
							cn.setBatchSize(1);
							cn.setInputLayer(new float[][][][]
							{ channelState });
							cn.propagation();
							if (n > 0 && i == 500)
								System.out.println(cn.getResultLayer()[0][0]
										+ " " + cn.getResultBatch()[0][0][0]);
							// ratio at 100_000 : (346 goalStates / 100_000
							// states)
							boolean same = Cube.solved(state);
							if (same)
							{
								total++;
								temp = 1f;
								break;
							}
							temp = Math.min(temp,
									Math.min(1f + cn.getResultLayer()[0][0],
											depth.get(i)));
						}
/*
						cn.setBatchSize(1);
						cn.setInputLayer(new float[][][][]
						{ Cube.toChannel(s) });
						cn.propagation();

						float alpha = (1f - 1f / depth.get(i)) * 0.1f;
						temp = alpha * cn.getResultLayer()[0][0]
								+ (1f - alpha) * temp;
*/
						y.add(new float[]
						{ temp });
						if ((i + 1) % 1000 == 0)
							System.out.print('.');

					}

					cn.setBatchSize(BATCH_SIZE);
					cn.setInputLayer(inputMatrix);
					outputMatrix = y.toArray(new float[][]
					{});
					cn.setOutputLayer(outputMatrix);
					System.out.println("] ok.\n[gS=" + total + "]");
					if (n > 0)
						System.out.println(y.get(0)[0] - 1f);
				}
			}
			if (t < THRESHOLD_ITERATIONS)
			{
				cn.train(n);

			}
		}

		System.out.println("Training done.");

		while (true)
		{
			int d = (int) (Math.random() * (8)) + 1;
			System.out.println(
					"Generating new example [d=" + d + "; t=" + t + "]");

			sc.setAnimation(false);
			sc.loadState(Cube.loadSolved());
			for (int i = 0; i < d; i++)
			{
				sc.rotate(RNotation.fromId(RNotation.randomRotation()));
			}
			state = sc.getState();
			sc.setAnimation(true);

			System.out.println("Starting pathfinder...");
			CPathFinder p = new CPathFinder(Cube.loadSolved(), cn);
			p.start(state);

			System.out.println(
					"Pathfinding done...path size " + p.getPath().size());
			for (int i = 0; i < p.getPath().size(); i++)
			{
				System.out.println(p.getPath().get(i));
				sc.addRotation(p.getPath().get(i));
				state = Cube.rotate(state, p.getPath().get(i));
				try
				{
					Thread.sleep(SpriteCube.MOVE_TIME + 1);
				}
				catch (InterruptedException e)
				{
					Thread.currentThread().interrupt();
				}
			}
			try
			{
				Thread.sleep(5000);
			}
			catch (InterruptedException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

	private void genStateSamples()
	{
		System.out.println("Generating state space from CPU...");
		System.out.print("[N=" + SAMPLE_SIZE + "; Depth=[1..K=" + K + "]]");

		for (int l = 0; l < SAMPLE_SIZE; l++)
		{
			int rd = 0;
			state = Cube.scramble(goalState,
					rd = (int) (Math.random() * K) + 1);
			X.add(Cube.toChannel(state));
			record.add(state);
			depth.add(rd);

		}
		;

		System.out.println(" ok.");
		inputMatrix = X.toArray(new float[][][][]
		{});

		cn.setInputLayer(inputMatrix);
	}

}
