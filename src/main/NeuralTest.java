package main;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import cube.Cube;
import cube.RNotation;
import math.Matrix;
import solver.NeuralNet;

public class NeuralTest {

	public static void cafe(Object o) {
		System.out.println(o);
	}

	public static void main(String[] args) {
		
		//main1(args);
		System.out.println(Matrix.invSqrt(5));
		System.out.println(1/Math.sqrt(5));
	}

	public static void main2(String[] args) {
		int[][][][] c = Cube.loadSolved();
		int S = 30;
		
		float[][] lInput = new float[S][Cube.FORMAT_SIZE_CUBE3];
		float[][] lOutput = new float[S][1];
		for (int n = 0; n < S; n++) {
			float depth = (int)(Math.random() * 5);
			int[][][][] t = Cube.copy(c);
			for (int a = 0; a < depth; a++) {
				t = Cube.rotate(t, RNotation.fromId(RNotation.randomRotation()));
			}
			lInput[n] = Cube.toCube3(t);
			System.out.println(depth);
			lOutput[n][0] = (float) (depth);
		}

		System.out.println("Expected output:");
		Matrix.print(lOutput);

		System.out.println("Training...");
		NeuralNet n = new NeuralNet(lInput[0].length, lOutput[0].length);
		n.setBatchSize(5);
		n.setInputLayer(lInput);
		n.setOutputLayer(lOutput);
		n.train(true);

		System.out.println("After training output:");
		//Matrix.print(n.getResultLayer());
		c = Cube.loadSolved();
		//Cube.rotate(c, RNotation.F);
		System.out.println();
		n.setBatchSize(1);
		n.setInputLayer(new float[][] { Cube.toCube3(c) });
		n.propagation();
		Matrix.print(n.getResultLayer());
		
		for (int i = 0; i < 1; i++)
			c = Cube.rotate(c, RNotation.fromId(i));
		n.setBatchSize(1);
		n.setInputLayer(new float[][] { Cube.toCube3(c) });
		n.propagation();
		Matrix.print(n.getResultLayer());
	}

	public static void main1(String[] args) {
		cafe(Math.sin(Math.toRadians(190)));

		int a = 6;
		float[][] lInput = new float[360 * a][1];
		float[][] lOutput = new float[360 * a][1];

		for (int i = 0; i < 360 * a; i++) {
			double rad = Math.toRadians((int) (Math.random() * (360 * a)) / (double) a);
			lInput[i][0] = (float) (rad);
			lOutput[i][0] = (float) ((Math.sin(rad) + 1) / 2f);// + 1) / 2f;
		}

		System.out.println("Expected output:");
		Matrix.print(lOutput);
//Matrix.print(lInput);
		System.out.println("Training...");
		NeuralNet n = new NeuralNet(lInput[0].length, lOutput[0].length);
		n.setBatchSize(36*a);
		n.setInputLayer(lInput);
		n.setOutputLayer(lOutput);
		n.train(true);
		// Matrix.print(n.getResultLayer());
		System.out.println("After training output:");
		for (double deg : new double[] { 90, 190, 20, 45, 60 }) {
			n.setBatchSize(1);
			double angle = (float) Math.toRadians(deg);
			n.setInputLayer(new float[][] { { (float) (angle) } });
			n.propagation();
			Matrix.print(n.getResultLayer());
			System.out.println((Math.sin(angle)+1f) / 2f);// + 1) / 2f);
		}
		/*
		 * int[][][][] c = Cube.loadSolved(); float[][] lInput = new
		 * float[12][Cube.FORMAT_SIZE_CUBE3]; for (int i = 0; i <
		 * RNotation.values().length; i++) { int[][][][] tc = c.clone(); Cube.rotate(tc,
		 * RNotation.fromId(i)); lInput[i] = Cube.toCube3(tc); }
		 * 
		 * float[][] lOutput = Matrix.loadIdentity(12);
		 * 
		 * System.out.println("Expected output:"); Matrix.print(lOutput);
		 * 
		 * System.out.println("Training..."); NeuralNetwork n = new
		 * NeuralNetwork(lInput, lOutput); n.train(true);
		 * 
		 * System.out.println("After training output:"); Matrix.print(n.getResult()); c
		 * = Cube.loadSolved(); Cube.rotate(c, RNotation.F); System.out.println();
		 * n.setInput(new float[][] { Cube.toCube3(c) }); Matrix.print(n.getResult());
		 */

	}
}
