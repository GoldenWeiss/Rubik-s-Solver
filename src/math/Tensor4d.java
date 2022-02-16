package math;

public class Tensor4d {

	public static float[][][] tSum(float[][][][] t) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++)
					for (int b = 0; b < e; b++)
						r[k][j][i] += t[b][k][j][i];
		return r;
	}

	public static void a_() {

	}

	public static float[][][] tMean(float[][][][] t) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][] r = new float[d][h][w];

		for (int k = 0; k < d; k++)
			for (int j = 0; j < h; j++)
				for (int i = 0; i < w; i++) {
					for (int b = 0; b < e; b++)
						r[k][j][i] += t[b][k][j][i];
					r[k][j][i] /= e;
				}

		return r;
	}

	public static float[][][][] square(float[][][][] t) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t[b][k][j][i] * t[b][k][j][i];

		return r;
	}

	public static float[][][][] substraction(float[][][][] t, float[][][] c) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t[b][k][j][i] - c[k][j][i];

		return r;
	}
	
	public static void substraction_(float[][][][] t, float[][][] c) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t[b][k][j][i] = t[b][k][j][i] - c[k][j][i];

	}

	public static float[][][][] substraction(float[][][][] t1, float[][][][] t2, float[][][] c) {
		int e = t1.length, d = t1[0].length, h = t1[0][0].length, w = t1[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t1[b][k][j][i] - t2[b][k][j][i] - c[k][j][i];

		return r;
	}
	
	public static void substraction_(float[][][][] t1, float[][][][] t2, float[][][] c) {
		int e = t1.length, d = t1[0].length, h = t1[0][0].length, w = t1[0][0][0].length;
		
		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t1[b][k][j][i] = t1[b][k][j][i] - t2[b][k][j][i] - c[k][j][i];

	}

	public static float[][][][] multiplication(float[][][][] t1, float[][][][] t2) {
		int e = t1.length, d = t1[0].length, h = t1[0][0].length, w = t1[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t1[b][k][j][i] * t2[b][k][j][i];

		return r;
	}

	public static float[][][][] multiplication(float[][][][] t, float[][][] c) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t[b][k][j][i] * c[k][j][i];

		return r;
	}

	public static void multiplication_(float[][][][] t, float[][][] c) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t[b][k][j][i] *= c[k][j][i];
	}
	
	public static void multiplication_(float[][][][] t, float[][][] c1, float[][][] c2) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t[b][k][j][i] *= c1[k][j][i] * c2[k][j][i];
	}

	public static float[][][][] multiply(float[][][][] t, float scalar) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		float[][][][] r = new float[e][d][h][w];

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						r[b][k][j][i] = t[b][k][j][i] * scalar;

		return r;
	}

	public static void multiply_(float[][][][] t, float scalar) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;
		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t[b][k][j][i] *= scalar;

	}

	public static void add(float[][][][] t, float[][][] c) {
		int e = t.length, d = t[0].length, h = t[0][0].length, w = t[0][0][0].length;

		for (int b = 0; b < e; b++)
			for (int k = 0; k < d; k++)
				for (int j = 0; j < h; j++)
					for (int i = 0; i < w; i++)
						t[b][k][j][i] += c[k][j][i];
	}
}
