package math;

import java.lang.reflect.Array;

public class Tensor3d {
	@FunctionalInterface
	public interface FloatTripleIntFunction {
		public float apply(int k, int j, int i);
	}

	/**
	 * 
	 * @param block
	 * @param dimensions from z to x
	 */
	public static float[][][] build(int d, int h, int w, FloatTripleIntFunction block) {
		float[][][] ary = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					ary[k][j][i] = block.apply(k, j, i);

		return ary;
	}

	public static void substract(float[][][] t1, float[][][] t2, float rate) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		if (d != t2.length || h != t2[0].length || w != t2[0][0].length)
			throw new IllegalArgumentException("tensor1 (" + d + "," + h + "," + w + ") dont match tensor2 " + t2.length
					+ "," + t2[0].length + "," + t2[0].length + "size!");
		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					t1[k][j][i] -= t2[k][j][i] * rate;
	}

	public static void add(float[][][] t1, float[][][] t2, float rate) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		if (d != t2.length || h != t2[0].length || w != t2[0][0].length)
			throw new IllegalArgumentException("tensor1 (" + d + "," + h + "," + w + ") dont match tensor2 " + t2.length
					+ "," + t2[0].length + "," + t2[0].length + "size!");
		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					t1[k][j][i] += t2[k][j][i] * rate;
	}

	public static void add(float[][][] t, float[][] m) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					t[k][j][i] += m[j][i];
	}

	public static float[][][] copy(float[][][] pTensor) {
		int d = pTensor.length, h = pTensor[0].length, w = pTensor[0][0].length;
		float[][][] copy = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					copy[k][j][i] = pTensor[k][j][i];

		return copy;
	}

	/**
	 * Vector matrix multiplication
	 * 
	 * @param t
	 */
	public static void multiply(float[][][] t, float[][] m) {
		for (int k = 0, d = t.length; k < d; k++)
			for (int j = 0, h = t[0].length; j < h; j++)
				for (int i = 0, w = t[0][0].length; i < w; i++)
					t[k][j][i] *= m[j][i];
	}

	public static float[][][] multiplication(float[][][] t1, float[][][] t2) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t1[k][j][i] * t2[k][j][i];

		return r;
	}
	
	public static float[][][] multiplication(float[][][] t1, float[][][] t2, float scalar) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t1[k][j][i] * t2[k][j][i] * scalar;

		return r;
	}

	public static float[][][] multiplication(float[][][] t, float[][] m) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t[k][j][i] * m[j][i];

		return r;

	}

	public static float[][] channelsSum(float[][][] t) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][] m = new float[h][w];

		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				for (int k = 0; k < d; k++)
					m[j][i] += t[k][j][i];

		return m;
	}

	public static float[][] channelsMean(float[][][] t) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][] m = new float[h][w];

		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++) {
				for (int k = 0; k < d; k++)
					m[j][i] += t[k][j][i];
				m[j][i] /= d;
			}

		return m;
	}

	public static float[][][] square(float[][][] t) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][][] sq = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					sq[k][j][i] = t[k][j][i] * t[k][j][i];

		return sq;

	}
	
	public static void square_(float[][][] t) {
		int d = t.length, h = t[0].length, w = t[0][0].length;

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					t[k][j][i] *= t[k][j][i];
	}
	
	public static float[][][] invsqrt(float[][][] t, float e) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][][] sq = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					sq[k][j][i] = (float) (1d / (Math.sqrt(t[k][j][i]+e))) ;

		return sq;

	}

	public static float[][][] substraction(float[][][] t, float[][] m) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][][] s = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					s[k][j][i] = t[k][j][i] - m[j][i];
		return s;
	}

	public static float[][][] addition(float[][][] t1, float[][][] t2) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		float[][][] s = new float[d][h][w];
		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					s[k][j][i] = t1[k][j][i] + t2[k][j][i];

		return s;
	}

	public static float[][][] substraction(float[][][] t1, float[][][] t2, float[][] m) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;

		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t1[k][j][i] - t2[k][j][i] - m[j][i];

		return r;
	}

	public static float[][][] multiply(float[][][] t, float scalar) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t[k][j][i] * scalar;

		return r;
	}

	public static float[][][] substraction(float[][][] t1, float[][][] t2) {
		int d = t1.length, h = t1[0].length, w = t1[0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					r[k][j][i] = t1[k][j][i] - t2[k][j][i];

		return r;
	}

	public static void multiply_(float[][][] t, float scalar) {
		int d = t.length, h = t[0].length, w = t[0][0].length;
		

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					t[k][j][i] = t[k][j][i] * scalar;
		
	}
}
