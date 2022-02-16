package solver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;

import cube.Cube;
import cube.RNotation;
import cube.SpriteCube;
import math.Matrix;

public class NeuralInterface implements Runnable {
	private static int[][][][] goalState = Cube.loadSolved();
	private static float[] cube3GoalState = Cube.toCube3(goalState);
	private static int TRAINING_ITERATIONS = 10_000;
	private static int SAMPLE_SIZE = 100_000;
	private static int BATCH_SIZE = 10_000;
	private static int K = Cube.GOD_NUMBER + 4;
	private static float LOSS_THRESHOLD = 0.05f;
	
	private static int CHECK_LOSS = 20;

	private NeuralNet net;
	private Map<float[], Integer> h; // 0:d, 1:w, 2:l
	private int[][][][] state;
	private float[] cube3State;
	private float[][] inputMatrix;
	private float[][] outputMatrix;
	public int n;
	private int t = -1;

	private SpriteCube sc;

	public NeuralInterface(SpriteCube sc) {
		this.sc = sc;
		h = new HashMap<>();
		net = new NeuralNet(Cube.FORMAT_SIZE_CUBE3, 1);
		net.setBatchSize(BATCH_SIZE);

	}

	public void run() {
		List<int[][][][]> record = new ArrayList<>();
		List<float[]> X = new ArrayList<>();
		List<float[]> y = new ArrayList<>();

		System.out.println("Generating state space from CPU...");
		System.out.print("[N=" + SAMPLE_SIZE + "; Depth=[1..K=" + K + "]]");
		for (int l = 0; l < SAMPLE_SIZE; l++) {

			state = Cube.scramble(goalState, (int) (Math.random() * K) + 1);
			X.add(Cube.toCube3(state));
			record.add(state);
		}
		System.out.println(" ok.");
		inputMatrix = list2dtoMatrix2d(X);
		net.setInputLayer(inputMatrix);

		for (n = 0; n < TRAINING_ITERATIONS; n++) {

			System.out.println("[n=" + n + "; MSE=" + net.getMSE() + "]");

			// X.clear();
			// record.clear();

			// avoid saturation if updated too fast
			if (n % CHECK_LOSS == 0 && net.getMSE() <= LOSS_THRESHOLD) {
				if (n > 0)
					Matrix.print(net.getResultLayer());
				System.out.println("Reached threshold [E=" + LOSS_THRESHOLD + "; t="+ ++t + "]");
				System.out.print("Diffusing new output to expected output...");
				y.clear();
				float temp;

				for (int[][][][] s : record) {
					temp = Float.MAX_VALUE;
					for (RNotation m : RNotation.values()) {

						state = Cube.rotate(s, m);
						cube3State = Cube.toCube3(state);
						
						net.setBatchSize(1);
						net.setInputLayer(new float[][] { cube3State });
						net.propagation();

						
						float x = 1f + (Arrays.equals(cube3GoalState, cube3State) ? 0f : net.getResultLayer()[0][0]);
						if ((x < temp || x == 1) && temp != 1)
							temp = x;
					}
					y.add(new float[] { temp });
				}
				net.setBatchSize(BATCH_SIZE);
				net.setInputLayer(inputMatrix);
				outputMatrix = list2dtoMatrix2d(y);
				net.setOutputLayer(outputMatrix);
				System.out.println(" ok.");
				
				
				
				//NeuralNet.GAMMA_LEARNING_RATE /= 2;
				//NeuralNet.BETA_LEARNING_RATE /= 2;
			}

			net.train(n);
			//NeuralNet.LEARNING_RATE *= 10;
			//Matrix.print(net.getResultLayer());
		}

		System.out.println("Training done.");
		while (true) {
			int d  = (int)(Math.random() * K) + 1;
			System.out.println("Generating new example [d="+d+"]");
			
			sc.setAnimation(false);
			sc.loadState(Cube.loadSolved());
			for (int i = 0; i < d; i++) {
				sc.rotate(RNotation.fromId(RNotation.randomRotation()));
			}
			state = sc.getState();
			sc.setAnimation(true);

			System.out.println("Starting pathfinder...");
			PathFinder p = new PathFinder(Cube.loadSolved(), net);
			p.start(state);

			System.out.println("Pathfinding done...path size " + p.getPath().size());
			for (int i = 0; i < p.getPath().size(); i++) {
				System.out.println(p.getPath().get(i));
				sc.addRotation(p.getPath().get(i));
				state = Cube.rotate(state, p.getPath().get(i));
				try {
					Thread.sleep(SpriteCube.MOVE_TIME + 1);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
			}
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	private float[][] list2dtoMatrix2d(List<float[]> list) {
		int s = list.size();
		float[][] mptr = new float[s][];
		for (int i = 0; i < s; i++) {
			mptr[i] = list.get(i);
		}
		return mptr;
	}

	private void printDepth(int[][][][] m) {
		int sum = 0;
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++)
					for (int c = 0; c < 3; c++)
						if (m[x][y][z][c] == goalState[x][y][z][c])
							sum++;
		System.out.println(sum / 81d);
	}
}
